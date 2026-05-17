"""
Group Basis MoE Learner — 群基组合持续学习器
=============================================

核心理念:
  概念 = Σ_g w_g · ( Σ_k β_{g,k} · Basis_{g,k}(z) )
  - 群权重 w 描述"哪种几何变换重要"
  - 基权重 β 描述"群内如何组合原子操作"
  - 扩展: 加新基原子 (K→K+1), 旧基冻结

训练流程 (继承 SEMA):
  Task 0: func phase → rd phase
  Task t: detect → expand (加新基) → train → freeze
"""

import logging, numpy as np, torch, torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from backbones.group_basis_moe import GroupBasisModules

num_workers = 8


class GroupBasisViTNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        from utils.inc_net import get_backbone
        self.backbone = get_backbone(args, pretrained)
        self.fc = None; self.args = args; self._device = args["device"][0]

    @property
    def feature_dim(self): return self.backbone.out_dim

    def forward(self, x):
        out = self.backbone(x); out.update({"logits": self.fc(out["features"])}); return out


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = GroupBasisViTNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.args = args
        self.lambda_rd = args.get("lambda_rd", 0.1)
        self.lambda_geo = args.get("lambda_geo", 0.01)
        self.lambda_sem = args.get("lambda_sem", 0.01)

    def after_task(self): self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.fc.bias)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1: self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            print(f"{sum(p.numel() for p in self._network.parameters()):,} total params")
            print(f"{sum(p.numel() for p in self._network.parameters() if p.requires_grad):,} trainable params")
            self._train_new(train_loader, test_loader)
            # Task 0 结束: 存储路由快照供后续截面保护
            self._collect_routing_snapshot(train_loader)
        else:
            # 检测阶段: 不保护 (需要自由路由来检测异常)
            for m in self._network.backbone.modules():
                if isinstance(m, GroupBasisModules): m.detecting_outlier = True
            dl = DataLoader(train_loader.dataset, batch_size=self.args.get("detect_batch_size", 128), shuffle=True, num_workers=num_workers)
            added = self._detect_outlier(dl, train_loader, test_loader, 0)
            for m in self._network.backbone.modules():
                if isinstance(m, GroupBasisModules): m.detecting_outlier = False
            if added == 0:
                # 训练阶段: 启用截面保护
                for m in self._network.backbone.modules():
                    if isinstance(m, GroupBasisModules): m.protect_old_sections()
                logging.info("No expansion — training with section protection")
                self.update_optimizer(num_epoch=self.args.get("func_epoch",5), lr=self.init_lr)
                self._init_train(self.args.get("func_epoch",5), train_loader, test_loader, self.optimizer, self.scheduler, "func")
                for m in self._network.backbone.modules():
                    if isinstance(m, GroupBasisModules): m.unprotect_sections()
        for m in self._network.backbone.modules():
            if isinstance(m, GroupBasisModules): m.end_of_task_training()
        # 更新下一任务的路由快照
        self._collect_routing_snapshot(train_loader)

    def _collect_routing_snapshot(self, loader):
        """在训练数据上计算平均路由分布, 存入各层 adapter 作为截面保护锚点。"""
        self._network.eval()
        # 开启捕获模式
        for m in self._network.backbone.modules():
            if isinstance(m, GroupBasisModules):
                m.start_snapshot_capture()
                m._training_rd = True
        with torch.no_grad():
            for i, (_, inputs, _) in enumerate(loader):
                if i >= 3: break
                inputs = inputs.to(self._device)
                _ = self._network(inputs)
        # 结束捕获, 存入 stored_group_weights / stored_basis_weights
        for m in self._network.backbone.modules():
            if isinstance(m, GroupBasisModules):
                m.finish_snapshot_capture()
                m._training_rd = False
        self._network.train()

    def _train_new(self, train_loader, test_loader):
        self.update_optimizer(num_epoch=self.args.get("func_epoch",5), lr=self.init_lr)
        self._init_train(self.args.get("func_epoch",5), train_loader, test_loader, self.optimizer, self.scheduler, "func")
        for m in self._network.backbone.modules():
            if isinstance(m, GroupBasisModules): m._training_rd = True
        self.update_rd_optimizer(num_epoch=self.args.get("rd_epoch",20), lr=self.args.get("rd_lr",0.01))
        self._init_train(self.args.get("rd_epoch",20), train_loader, test_loader, self.rd_optimizer, self.rd_scheduler, "rd")
        for m in self._network.backbone.modules():
            if isinstance(m, GroupBasisModules): m._training_rd = False

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        is_added = False
        for _, inputs, targets in detect_loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            outcome = self._network(inputs)
            if sum(outcome["added_record"]) > 0:
                added += 1; is_added = True
                for m in self._network.backbone.modules():
                    if isinstance(m, GroupBasisModules): m.detecting_outlier = False
                self._train_new(train_loader, test_loader)
                for m in self._network.backbone.modules():
                    if isinstance(m, GroupBasisModules): m.freeze_functional(); m.freeze_rd(); m.reset_newly_added_status()
                for m in self._network.backbone.modules():
                    if isinstance(m, GroupBasisModules): m.detecting_outlier = True
        return self._detect_outlier(detect_loader, train_loader, test_loader, added) if is_added else added

    def _compute_geo_loss(self):
        loss = torch.tensor(0., device=self._device)
        for m in self._network.backbone.modules():
            if isinstance(m, GroupBasisModules):
                for a in m.adapters: loss = loss + a.orthogonality_error()
        return loss

    def _init_train(self, total_epoch, train_loader, test_loader, optimizer, scheduler, phase="func"):
        tracker = self.get_loss_tracker(); prog_bar = tqdm(range(total_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train(); correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outcome = self._network(inputs); logits = outcome["logits"][:, :self._total_classes]
                if self._cur_task > 0: logits[:, :self._known_classes] = -float("inf")
                if phase == "func":
                    loss = F.cross_entropy(logits, targets)
                    # Task 0: 先学分类, 不加几何约束 (SO/LR 正则化延迟到 rd 阶段)
                    # Task t: 加入几何约束防止过拟合
                    if self._cur_task > 0:
                        loss = loss + self.lambda_rd * outcome.get("rd_loss", torch.tensor(0., device=self._device)) + self.lambda_geo * self._compute_geo_loss()
                else:
                    loss = outcome.get("rd_loss", torch.tensor(0., device=self._device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                tracker.update(**{phase: loss.item()})
                _, preds = torch.max(logits, 1); correct += preds.eq(targets.expand_as(preds)).cpu().sum(); total += len(targets)
            scheduler.step()
            ta = np.around(tensor2numpy(correct)*100/total, 2); al = tracker.flush(epoch)
            tst = self._compute_accuracy(self._network, test_loader)
            prog_bar.set_description(f"{phase} Task {self._cur_task}, Epoch {epoch+1}/{total_epoch} => Loss {al.get(phase,0):.3f}, Train {ta:.2f}, Test {tst:.2f}")
        logging.info(prog_bar.desc)

    def _eval_cnn(self, loader):
        self._network.eval(); yp, yt = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad(): logits = self._network(inputs)["logits"][:, :self._total_classes]
            yp.append(torch.topk(logits, k=self.topk, dim=1)[1].cpu().numpy()); yt.append(targets.cpu().numpy())
        return np.concatenate(yp), np.concatenate(yt)

    def _compute_accuracy(self, model, loader):
        model.eval(); c, t = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad(): logits = model(inputs)["logits"][:, :self._total_classes]
            c += (torch.max(logits, 1)[1].cpu() == targets).sum(); t += len(targets)
        return np.around(tensor2numpy(c)*100/t, 2)

    def update_optimizer(self, num_epoch=20, lr=None):
        lr = self.init_lr if lr is None else lr
        params = [p for n, p in self._network.named_parameters() if p.requires_grad and ("router" in n or "fc" in n or "vpt" in n or "basis_bank" in n or "group_ae" in n or "gamma" in n or "down_proj" in n or "up_proj" in n)]
        if not params: params = [p for n, p in self._network.named_parameters() if p.requires_grad]
        o = self.args.get("optimizer","sgd"); wd = self.args.get("weight_decay",0.0005)
        self.optimizer = optim.SGD(params, momentum=0.9, lr=lr, weight_decay=wd) if o=="sgd" else optim.AdamW(params, lr=lr, weight_decay=wd)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epoch, eta_min=self.args.get("min_lr",1e-8))

    def update_rd_optimizer(self, num_epoch=20, lr=None):
        lr = self.args.get("rd_lr",0.01) if lr is None else lr
        params = [p for n, p in self._network.named_parameters() if ("group_ae" in n) and p.requires_grad]
        if not params: params = [p for n, p in self._network.named_parameters() if "group_ae" in n]
        o = self.args.get("optimizer","sgd"); wd = self.args.get("weight_decay",0.0005)
        self.rd_optimizer = optim.SGD(params, momentum=0.9, lr=lr, weight_decay=wd) if o=="sgd" else optim.AdamW(params, lr=lr, weight_decay=wd)
        self.rd_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.rd_optimizer, T_max=num_epoch, eta_min=self.args.get("min_lr",1e-8))

    def save_checkpoint(self, fn):
        sd = self._network.state_dict()
        torch.save({k:v for k,v in sd.items() if "adapter" in k or "fc" in k or "basis" in k}, f"{fn}.pth")

    def load_checkpoint(self, fn): self._network.load_state_dict(torch.load(fn), strict=False)
