# -*- coding: utf-8 -*-
# 修复版训练脚本：SEAM+DSCA 真正参与训练，训练结束自动跑测试集评估
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer


# ===================== 自定义模块（与原版完全一致，勿改） =====================
class SEAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x.weight = nn.Parameter(
            torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32),
            requires_grad=False
        )
        self.sobel_y.weight = nn.Parameter(
            torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32),
            requires_grad=False
        )
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels)
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.edge_enhance[-2].weight, 0.0)
        nn.init.constant_(self.edge_enhance[-2].bias, -10.0)

    def forward(self, x):
        gray = torch.mean(x, dim=1, keepdim=True)
        edge_x = self.sobel_x(gray)
        edge_y = self.sobel_y(gray)
        edge_feat = torch.cat([edge_x, edge_y], dim=1)
        edge_att = self.sigmoid(self.edge_enhance(edge_feat))
        return x + x * edge_att


class DSCA(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.spatial_reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        nn.init.constant_(self.spatial_reduce[-2].weight, 0.0)
        nn.init.constant_(self.spatial_reduce[-2].bias, -10.0)
        nn.init.constant_(self.channel_reduce[-2].weight, 0.0)
        nn.init.constant_(self.channel_reduce[-2].bias, -10.0)

    def forward(self, x):
        spatial_att = self.spatial_reduce(x)
        channel_att = self.channel_reduce(x)
        out = x * spatial_att * channel_att
        return out + x


# ===================== 在 nc=3 的 DetectionModel 上做手术 =====================
def insert_modules(det_model):
    new_layers = list(det_model.model.children())
    insert_config = [(16, 128), (19, 256), (22, 512)]
    for ori_idx, channels in sorted(insert_config, reverse=True):
        insert_pos = ori_idx + 1
        ea = SEAM(channels=channels)
        sc = DSCA(channels=channels)
        ea.f = -1   # 取上一层输出
        sc.f = -1
        new_layers.insert(insert_pos, sc)
        new_layers.insert(insert_pos, ea)
    det_model.model = nn.Sequential(*new_layers)

    # 重排每层索引 i，并保证每层都有 f 属性（ultralytics 前向依赖）
    for idx, m in enumerate(det_model.model):
        m.i = idx
        if not hasattr(m, "f"):
            m.f = -1

    detect_layer = det_model.model[-1]
    detect_layer.f = [18, 23, 28]
    detect_layer.i = len(det_model.model) - 1

    # 按新的连接关系重算 save 列表（Detect 需要的层输出必须被缓存）
    needed = set()
    for m in det_model.model:
        f = m.f
        if isinstance(f, (list, tuple)):
            needed.update(v for v in f if v != -1)
        elif f != -1:
            needed.add(f)
    det_model.save = sorted(needed)

    print("\n----- 插入后层索引（应看到 SEAM/DSCA，Detect 在 29 层） -----")
    for idx, m in enumerate(det_model.model):
        print(f"layer[{idx}] : {type(m).__name__}")
    print("Detect.f =", detect_layer.f, " save =", det_model.save)
    print("--------------------------------------------------------------\n")
    return det_model


if __name__ == "__main__":
    # 1) 构建 nc=3 的 stock yolo11s（等价于原 train() 内部 get_model 的行为）
    pretrained = YOLO("yolo11s.pt")
    model = DetectionModel(cfg=pretrained.model.yaml, ch=3, nc=3, verbose=True)
    model.load(pretrained.model)  # backbone/neck 迁移，head 因类别数不同随机初始化

    # 2) 插入 SEAM+DSCA
    model = insert_modules(model)

    # 3) 直接把 nn.Module 塞给 DetectionTrainer（setup_model 检测到是 nn.Module 就不会重建）
    trainer = DetectionTrainer(overrides=dict(
        model="yolo11s.pt",
        data=r"E:\YOLO\data.yaml",
        epochs=120,          # 正式训练 120 轮
        imgsz=640,
        batch=8,
        workers=0,
        device=0,
        amp=False,
        patience=40,
        optimizer="AdamW",
        lr0=0.0006,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=4,
        warmup_bias_lr=0.05,
        warmup_momentum=0.8,
        cos_lr=True,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=10,
        auto_augment="randaugment",
        mosaic=1.0,
        fliplr=0.5,
        erasing=0.25,
        translate=0.05,
        save=True,
        save_period=1,
        plots=True,
        verbose=True,
        exist_ok=False,
        name="train_120epoch_EANet_SCConv",
    ))
    trainer.model = model   # ★ 关键：绕过 yaml 重建
    trainer.train()

    # 4) 训练结束自动跑测试集评估（即之前给你的那段 val 代码）
    best = trainer.save_dir / "weights" / "best.pt"
    test_model = YOLO(str(best))
    metrics = test_model.val(
        data=r"E:\YOLO\data.yaml",
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        plots=True,
        project="runs/val",
        name="test_eval_fixed",
        exist_ok=True,
    )
    print(f"\n[测试集] 总体 mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
