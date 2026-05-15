"""
Geometry-SEMA Learner (Group-MoE + Shared MambaFlow)
=====================================================

实现 suggest.md section 11-13 的训练流程。

三层损失系统 (section 11):
  L_total = L_cls + lambda_geo_rd * L_geo_rd + lambda_sem * L_sem

其中:
  L_cls = CE(logits, y)                                    -- 分类损失

  L_geo_rd = alpha_rd * L_group_RD                         -- 群感知 RD 损失
           + alpha_geo * L_geo                             -- SO 正交约束
           + alpha_bal * L_balance                         -- 路由平衡

  L_sem = L_proto                                          -- 原型一致性
        + beta_router * L_router                           -- 路由一致性
        + (可选) beta_flow * L_flow                         -- 流一致性

自适应权重 (section 12):
  alpha_rd  = clip(1 + gamma_z * z_norm, min, max)         -- 高异常时增大 RD 权重
  alpha_geo = clip(alpha0 * (1 + gamma_o * orth_error))    -- 高非正交时增大几何约束
  alpha_bal = clip(alpha0 * (1 + gamma_u * usage_imbalance)) -- 不平衡时增大平衡约束

训练流程 (section 13):
  Task 0:
    1. 构建预训练 ViT 主干
    2. 初始化固定 GroupBank (每群 1 个专家)
    3. 初始化共享 MambaFlow
    4. 训练分类路径 (Group-MoE 专家 + 路由器 + 分类器)
    5. 训练 Group-Aware AE/RD
    6. 任务结束时冻结旧专家和 RD 统计

  Task t > 0:
    1. 启用 detecting_outlier
    2. 检测: 计算群概率和 z-score, 判断是否需要扩展
    3. 无扩展: 微调路由和允许的参数
    4. 有扩展: 添加群特定专家 -> 训练新专家 + 路由器 + RD
    5. 任务结束: 冻结旧专家 + 冻结 RD 统计 + 存储路由概率
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from backbones.sema_geometry_moe import GeometrySEMAModules

num_workers = 8


class GeoSEMAVitNet(nn.Module):
    """Geometry-SEMA ViT 网络包装器。

    封装 backbone (VisionTransformer) + 分类头 fc,
    提供统一的 forward 接口返回字典格式输出。

    与标准 SEMAVitNet 的区别:
      - backbone 返回字典 (含 group_rd_loss, group_probs 等)
      - fc 需要从 backbone 的 features 生成 logits
    """

    def __init__(self, args, pretrained):
        super().__init__()
        from utils.inc_net import get_backbone
        self.backbone = get_backbone(args, pretrained)
        self.fc = None
        self.args = args
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        """提取特征向量 (用于评估)。"""
        return self.backbone(x)

    def forward(self, x):
        """前向传播: 特征提取 + 分类。

        Returns:
            dict: 包含 logits, features, group_rd_loss, added_record,
                  all_group_probs, all_z_scores
        """
        out = self.backbone(x)
        x_feat = out["features"]
        out.update({"logits": self.fc(x_feat)})
        return out


class Learner(BaseLearner):
    """Geometry-SEMA 持续学习器。

    实现基于 Group-MoE 的持续学习:
      - 群特定专家扩展 (而非适配器级别)
      - 共享 MambaFlow (不扩展)
      - 三层损失训练 (分类 + 几何-RD + 语义保持)
      - 自适应损失权重 (基于几何状态)

    核心属性:
      - lambda_geo_rd: L_geo_rd 权重
      - lambda_sem:    L_sem 权重
      - alpha_rd/geo/bal: L_geo_rd 内部自适应权重
      - old_router_probs: 存储旧任务的路由概率 (用于语义保持)
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = GeoSEMAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.args = args

        # ── 三层损失权重 ──
        self.lambda_geo_rd = args.get("lambda_geo_rd", 0.1)  # 几何-RD 损失
        self.lambda_sem = args.get("lambda_sem", 0.01)        # 语义保持损失
        self.beta_router = args.get("beta_router", 0.01)      # 路由一致性权重

        # ── L_geo_rd 内部自适应权重 ──
        self.alpha_rd = args.get("alpha_rd", 1.0)             # 群 RD 损失
        self.alpha_geo = args.get("alpha_geo", 0.1)           # SO 正交约束
        self.alpha_bal = args.get("alpha_bal", 0.01)          # 路由平衡

        # ── 自适应权重范围 ──
        self.alpha_rd_min = args.get("alpha_rd_min", 0.1)
        self.alpha_rd_max = args.get("alpha_rd_max", 10.0)
        self.alpha_geo_min = args.get("alpha_geo_min", 0.01)
        self.alpha_geo_max = args.get("alpha_geo_max", 1.0)
        self.alpha_bal_min = args.get("alpha_bal_min", 0.001)
        self.alpha_bal_max = args.get("alpha_bal_max", 0.1)

        # ── 自适应权重调节系数 ──
        self.gamma_z = args.get("gamma_z", 0.5)               # z-score 调节
        self.gamma_o = args.get("gamma_o", 1.0)               # 正交误差调节
        self.gamma_u = args.get("gamma_u", 1.0)               # 不平衡调节

        # ── 语义保持状态 ──
        self.old_router_probs = None   # 旧任务路由概率 (用于 L_router)
        self.old_prototypes = None     # 旧任务类原型 (用于 L_proto)

    def after_task(self):
        """任务结束后更新已知类别数。"""
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        """增量训练主循环。

        任务 0: 初始训练 (func + rd 两阶段)
        任务 t: 检测 + 扩展 + 训练
        """
        self._cur_task += 1

        # ── 任务 0: 初始化分类头 ──
        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.fc.bias)

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes))

        # ── 数据加载 ──
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=num_workers)
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers)

        # ── 多 GPU 支持 ──
        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        """训练主逻辑: 分发到任务 0 或任务 t 流程。"""
        self._network.to(self._device)

        if self._cur_task == 0:
            # ── 任务 0: 完整两阶段训练 ──
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f"{total_params:,} total parameters.")
            total_trainable = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f"{total_trainable:,} training parameters.")

            # 两阶段训练: func phase + rd phase
            self._train_new(train_loader, test_loader)

            # 存储路由概率供后续任务语义保持
            self._store_router_probs(train_loader)
        else:
            # ── 任务 t > 0: 检测 + 扩展 + 训练 ──

            # Step 1: 启用异常检测模式
            for module in self._network.backbone.modules():
                if isinstance(module, GeometrySEMAModules):
                    module.detecting_outlier = True

            # Step 2: 检测并扩展
            detect_loader = DataLoader(
                train_loader.dataset,
                batch_size=self.args.get("detect_batch_size", 128),
                shuffle=True, num_workers=num_workers)
            added = self._detect_outlier(detect_loader, train_loader, test_loader, 0)

            # Step 3: 关闭检测模式
            for module in self._network.backbone.modules():
                if isinstance(module, GeometrySEMAModules):
                    module.detecting_outlier = False

            # Step 4: 无扩展时微调路由器
            if added == 0:
                logging.info("No expansion — fine-tuning routers only")
                self.update_optimizer_and_scheduler(
                    num_epoch=self.args.get("func_epoch", 5), lr=self.init_lr)
                self._init_train(self.args.get("func_epoch", 5),
                                 train_loader, test_loader,
                                 self.optimizer, self.scheduler, phase="func")

        # ── 任务结束: 冻结旧专家 + 存储路由概率 ──
        for module in self._network.backbone.modules():
            if isinstance(module, GeometrySEMAModules):
                module.end_of_task_training()

        self._store_router_probs(train_loader)

    def _train_new(self, train_loader, test_loader):
        """任务 0 两阶段训练: func phase -> rd phase。

        func phase: 训练分类路径 (专家 + 路由器 + 分类器)
        rd phase:  训练 Group-Aware AE/RD (表征描述器)
        """
        # ── Phase 1: 功能训练 ──
        self.update_optimizer_and_scheduler(
            num_epoch=self.args.get("func_epoch", 5), lr=self.init_lr)
        self._init_train(self.args.get("func_epoch", 5),
                         train_loader, test_loader,
                         self.optimizer, self.scheduler, phase="func")

        # ── Phase 2: RD 训练 ──
        self.update_rd_optimizer_and_scheduler(
            num_epoch=self.args.get("rd_epoch", 20), lr=self.args.get("rd_lr", 0.01))
        self._init_train(self.args.get("rd_epoch", 20),
                         train_loader, test_loader,
                         self.rd_optimizer, self.rd_scheduler, phase="rd")

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        """异常检测 + 递归扩展。

        检测逻辑 (section 9):
          对每个 batch:
            前向传播 -> 检查 added_record
            若任何深层触发扩展:
              关闭检测 -> 训练新专家 -> 冻结 -> 重新开启检测 (递归)
            若未触发:
              继续下一个 batch

        Args:
            detect_loader: 检测数据加载器
            train_loader:  训练数据加载器
            test_loader:   测试数据加载器
            added:         已扩展计数

        Returns:
            int: 总扩展次数
        """
        is_added = False

        for i, (_, inputs, targets) in enumerate(detect_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            model_outcome = self._network(inputs)
            added_record = model_outcome["added_record"]

            # 检查是否触发扩展
            if sum(added_record) > 0:
                added += 1
                is_added = True

                # 关闭检测模式 -> 训练新专家
                for module in self._network.backbone.modules():
                    if isinstance(module, GeometrySEMAModules):
                        module.detecting_outlier = False

                self._train_new(train_loader, test_loader)

                # 冻结新专家 -> 重新开启检测
                for module in self._network.backbone.modules():
                    if isinstance(module, GeometrySEMAModules):
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()

                for module in self._network.backbone.modules():
                    if isinstance(module, GeometrySEMAModules):
                        module.detecting_outlier = True

        # 递归: 可能触发多次扩展
        if is_added:
            return self._detect_outlier(
                detect_loader, train_loader, test_loader, added)
        return added

    # ═══════════════════════════════════════════════════════════════════
    # 三层损失计算
    # ═══════════════════════════════════════════════════════════════════

    def _compute_adaptive_weights(self, outcome):
        """计算自适应损失权重 (section 12)。

        根据当前几何状态动态调整 L_geo_rd 的内部权重:

        alpha_rd  = clip(1 + gamma_z * z_norm, min, max)
          - 高异常 (大 z-score) 时增大 RD 权重, 加强分布外检测

        alpha_geo = clip(alpha0 * (1 + gamma_o * orth_error), min, max)
          - 高非正交误差时增大几何约束, 加强 SO 正则化

        alpha_bal = clip(alpha0 * (1 + gamma_u * usage_imbalance), min, max)
          - 群使用不平衡时增大平衡约束, 鼓励均匀路由

        所有计算使用 stopgrad 防止模型博弈自适应权重。

        Args:
            outcome: 模型输出字典

        Returns:
            tuple: (alpha_rd, alpha_geo, alpha_bal) 自适应权重
        """
        # ── 计算 z-score 统计 ──
        all_z_scores = outcome.get("all_z_scores", [])
        z_norm = 0.0
        if all_z_scores:
            z_stack = torch.cat([zs.mean(dim=0) for zs in all_z_scores])
            z_norm = z_stack.norm().item()

        # ── 计算正交误差 ──
        orth_error = 0.0
        for module in self._network.backbone.modules():
            if isinstance(module, GeometrySEMAModules):
                for adapter in module.adapters:
                    orth_error += adapter.orthogonality_error().item()

        # ── 计算群使用不平衡度 ──
        usage_imbalance = 0.0
        all_group_probs = outcome.get("all_group_probs", [])
        if all_group_probs:
            mean_probs = torch.stack(
                [gp.mean(dim=0) for gp in all_group_probs]
            ).mean(dim=0)  # [G]
            # 与均匀分布的 KL 散度
            G = len(mean_probs)
            prior = torch.ones(G, device=mean_probs.device) / G
            usage_imbalance = torch.sum(
                mean_probs.detach() * (torch.log(mean_probs.detach() + 1e-8) - torch.log(prior))
            ).item()

        # ── 自适应权重 (stopgrad 保证稳定性) ──
        alpha_rd = np.clip(
            1.0 + self.gamma_z * z_norm,
            self.alpha_rd_min, self.alpha_rd_max
        )
        alpha_geo = np.clip(
            self.alpha_geo * (1.0 + self.gamma_o * orth_error),
            self.alpha_geo_min, self.alpha_geo_max
        )
        alpha_bal = np.clip(
            self.alpha_bal * (1.0 + self.gamma_u * usage_imbalance),
            self.alpha_bal_min, self.alpha_bal_max
        )

        return alpha_rd, alpha_geo, alpha_bal

    def _compute_geo_rd_loss(self, outcome):
        """计算几何-RD 损失 (section 11.2)。

        L_geo_rd = alpha_rd * L_group_RD + alpha_geo * L_geo + alpha_bal * L_balance

        三项的含义:
          - L_group_RD: 群感知重建误差, 检测分布偏移
          - L_geo: SO 正交约束, 保持旋转专家的几何性质
          - L_balance: 路由平衡, 鼓励所有群被均等使用

        Args:
            outcome: 模型输出字典

        Returns:
            geo_rd_loss: scalar 加权几何-RD 损失
            components:  dict 各项损失的标量值 (用于日志)
        """
        # ── 自适应权重 ──
        alpha_rd, alpha_geo, alpha_bal = self._compute_adaptive_weights(outcome)

        # ── L_group_RD: 群感知 RD 损失 ──
        group_rd_loss = outcome.get("group_rd_loss",
                                    torch.tensor(0., device=self._device))

        # ── L_geo: SO 正交约束 ──
        # ||R^T R - I||^2 对所有 SO 专家求和
        ortho_error = torch.tensor(0., device=self._device)
        for module in self._network.backbone.modules():
            if isinstance(module, GeometrySEMAModules):
                for adapter in module.adapters:
                    ortho_error = ortho_error + adapter.orthogonality_error()

        # ── L_balance: 路由平衡 ──
        # KL(batch_mean_p || uniform), 鼓励均等的群使用
        balance_loss = torch.tensor(0., device=self._device)
        all_group_probs = outcome.get("all_group_probs", [])
        if all_group_probs:
            # 跨深层平均的群概率分布
            mean_probs = torch.stack(
                [gp.mean(dim=0) for gp in all_group_probs]
            ).mean(dim=0)  # [G]
            G = len(mean_probs)
            prior = torch.ones(G, device=mean_probs.device) / G
            # KL(batch_dist || uniform)
            balance_loss = torch.sum(
                mean_probs * (torch.log(mean_probs + 1e-8) - torch.log(prior))
            )

        # ── 加权求和 ──
        geo_rd = (
            alpha_rd * group_rd_loss
            + alpha_geo * ortho_error
            + alpha_bal * balance_loss
        )

        return geo_rd, {
            "group_rd": group_rd_loss.item(),
            "ortho": ortho_error.item(),
            "balance": balance_loss.item(),
            "alpha_rd": alpha_rd,
            "alpha_geo": alpha_geo,
            "alpha_bal": alpha_bal,
        }

    def _compute_sem_loss(self, outcome):
        """计算语义保持损失 (section 11.3)。

        L_sem = L_proto + beta_router * L_router + (可选) beta_flow * L_flow

        语义保持策略:
          - 无旧样本存储 -> 使用类原型 + RD 统计 + 合成原型 token
          - L_proto: 新旧类原型一致性, 防止表征漂移
          - L_router: 新旧路由概率 KL 散度, 防止路由漂移
          - L_flow: MambaFlow 对原型序列的输出一致性 (可选)

        Args:
            outcome: 模型输出字典

        Returns:
            sem_loss: scalar 语义保持损失
        """
        proto_loss = torch.tensor(0., device=self._device)
        router_loss = torch.tensor(0., device=self._device)
        flow_loss = torch.tensor(0., device=self._device)

        # ── L_router: 路由一致性 ──
        # KL(p_old(g|h) || p_new(g|h))
        # 鼓励新旧任务的路由行为一致, 防止路由器遗忘
        if self.old_router_probs is not None:
            all_group_probs = outcome.get("all_group_probs", [])
            if all_group_probs and len(self.old_router_probs) > 0:
                for i, (new_probs, old_probs) in enumerate(
                    zip(all_group_probs, self.old_router_probs)):
                    if i >= len(self.old_router_probs):
                        break
                    # KL(old || new): 鼓励新路由接近旧路由
                    kl = torch.sum(
                        old_probs.to(self._device)
                        * (torch.log(old_probs.to(self._device) + 1e-8)
                           - torch.log(new_probs + 1e-8))
                    )
                    router_loss = router_loss + kl
                router_loss = router_loss / max(len(all_group_probs), 1)

        # ── L_proto: 原型一致性 ──
        # sum_k ||c_k^t - c_k^{t-1}||^2
        # 维护旧类别的特征原型, 防止表征漂移
        # (需要外部原型存储, 当前为占位)

        # ── L_flow: 流一致性 (可选) ──
        # ||MambaFlow_t(proto_seq_old) - MambaFlow_{t-1}(proto_seq_old)||^2
        # (需要外部原型序列存储, 当前为占位)

        sem_loss = proto_loss + self.beta_router * router_loss + flow_loss
        return sem_loss

    def _init_train(self, total_epoch, train_loader, test_loader,
                    optimizer, scheduler, phase="func"):
        """统一的训练循环。

        支持两个阶段:
          - phase="func": 完整三层损失训练 (L_cls + L_geo_rd + L_sem)
          - phase="rd":   仅 RD 损失训练 (L_group_RD)

        Args:
            total_epoch:  训练 epoch 数
            train_loader: 训练数据加载器
            test_loader:  测试数据加载器
            optimizer:    优化器
            scheduler:    学习率调度器
            phase:        训练阶段 ("func" | "rd")
        """
        tracker = self.get_loss_tracker()
        prog_bar = tqdm(range(total_epoch))

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outcome = self._network(inputs)

                logits = outcome["logits"]
                logits = logits[:, :self._total_classes]

                # 只允许分类到已知类别范围
                if self._cur_task > 0:
                    logits[:, :self._known_classes] = -float("inf")

                if phase == "func":
                    # ── L_cls: 交叉熵分类损失 ──
                    loss_cls = F.cross_entropy(logits, targets)

                    # ── L_geo_rd: 几何-RD 损失 ──
                    loss_geo_rd, geo_components = self._compute_geo_rd_loss(outcome)

                    # ── L_sem: 语义保持损失 ──
                    loss_sem = self._compute_sem_loss(outcome)

                    # ── 总损失 ──
                    loss = (
                        loss_cls
                        + self.lambda_geo_rd * loss_geo_rd
                        + self.lambda_sem * loss_sem
                    )
                elif phase == "rd":
                    # ── RD 阶段: 仅群感知 RD 损失 ──
                    loss = outcome.get("group_rd_loss",
                                       torch.tensor(0., device=self._device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tracker.update(**{phase: loss.item()})

                # 准确率统计
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_losses = tracker.flush(epoch)
            avg_loss = avg_losses.get(phase, 0)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = (
                "{} Task {}, Epoch {}/{} => Loss {:.3f}, "
                "Train_accy {:.2f}, Test_accy {:.2f}".format(
                    phase, self._cur_task, epoch + 1, total_epoch,
                    avg_loss, train_acc, test_acc))
            prog_bar.set_description(info)

        logging.info(info)

    def _store_router_probs(self, loader):
        """存储当前任务的路由概率分布。

        用于下一任务的语义保持损失 (L_router):
          在新任务中, KL(old_probs || new_probs) 鼓励路由器保持一致行为。

        Args:
            loader: 数据加载器 (2 个 batch 足够估计分布)
        """
        self._network.eval()
        all_probs = []
        with torch.no_grad():
            for i, (_, inputs, _) in enumerate(loader):
                if i >= 2:  # 2 个 batch 足够
                    break
                inputs = inputs.to(self._device)
                outcome = self._network(inputs)
                gp = outcome.get("all_group_probs", [])
                if gp:
                    # 跨 batch 平均
                    avg = torch.stack([p.mean(dim=0) for p in gp])
                    all_probs.append(avg)

        if all_probs:
            # 跨深层存储: 每层一个平均分布
            self.old_router_probs = [
                torch.stack([ap[i] for ap in all_probs]).mean(dim=0)
                for i in range(len(all_probs[0]))
            ]
        else:
            self.old_router_probs = None

    # ═══════════════════════════════════════════════════════════════════
    # 评估
    # ═══════════════════════════════════════════════════════════════════

    def _eval_cnn(self, loader):
        """评估: 返回预测和标签。"""
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_accuracy(self, model, loader):
        """计算 top-1 准确率。"""
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = model(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    # ═══════════════════════════════════════════════════════════════════
    # 优化器管理
    # ═══════════════════════════════════════════════════════════════════

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        """创建/更新 func phase 优化器。

        优化目标: 所有可训练参数 (投影层, GroupBank, MambaFlow, 路由器, 分类头)
        """
        lr = self.args["init_lr"] if lr is None else lr
        func_params = [
            p for n, p in self._network.named_parameters()
            if ("functional" in n or "router" in n or "fc" in n or "vpt" in n
                or "down_proj" in n or "up_proj" in n or "gamma" in n
                or "group_bank" in n or "group_ae" in n
                or "mamba_flow" in n)
            and p.requires_grad
        ]
        if self.args.get("optimizer", "sgd") == "sgd":
            self.optimizer = optim.SGD(
                func_params, momentum=0.9, lr=lr,
                weight_decay=self.args.get("weight_decay", 0.0005))
        elif self.args.get("optimizer") == "adam":
            self.optimizer = optim.AdamW(
                func_params, lr=lr,
                weight_decay=self.args.get("weight_decay", 0.0005))

        min_lr = self.args.get("min_lr", 1e-8)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epoch, eta_min=min_lr)

    def update_rd_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        """创建/更新 RD phase 优化器。

        RD 阶段只训练 Group-Aware AE/RD 参数。
        """
        lr = self.args.get("rd_lr", 0.01) if lr is None else lr
        rd_params = [
            p for n, p in self._network.named_parameters()
            if ("group_ae" in n or "rd" in n)
            and p.requires_grad
        ]
        if self.args.get("optimizer", "sgd") == "sgd":
            self.rd_optimizer = optim.SGD(
                rd_params, momentum=0.9, lr=lr,
                weight_decay=self.args.get("weight_decay", 0.0005))
        elif self.args.get("optimizer") == "adam":
            self.rd_optimizer = optim.AdamW(
                rd_params, lr=lr,
                weight_decay=self.args.get("weight_decay", 0.0005))

        min_lr = self.args.get("min_lr", 1e-8)
        self.rd_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.rd_optimizer, T_max=num_epoch, eta_min=min_lr)

    def save_checkpoint(self, filename):
        """保存检查点 (仅适配器和分类头参数)。"""
        state_dict = self._network.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if ("adapter" in k or "fc" in k and "block" not in k
                    or "group" in k or "down_proj" in k or "up_proj" in k
                    or "mamba_flow" in k):
                save_dict[k] = v
        torch.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        """加载检查点。"""
        self._network.load_state_dict(torch.load(filename), strict=False)
