---
date: 2026-02-27
keywords: [OccFormer, 3D Occupancy, Dual-path Transformer, Mask2Former, ICCV 2023]
tags: [Level-02, Perception-Hardcore, Occupancy-Transformer]
---

# OccFormer-论文速读

## 0. 基本信息
- **发表时间**: 2023年 (ICCV 2023)
- **作者单位**: 上海人工智能实验室 (Shanghai AI Lab)
- **代码仓库**: [https://github.com/zhangyp15/OccFormer](https://github.com/zhangyp15/OccFormer)
- **Tags**: #3D语义占据 #Transformer占据 #Dual-path架构 #ICCV2023

---

## 1. 🔪 今日锐评
> **OccFormer** 解决了 3D 占据感知的“既要也要”难题。
> 
> **核心洞察**：Occupancy 任务面临着全局结构（哪里是路）和局部细节（那根电线杆在哪）的矛盾。OccFormer 提出了 **Dual-path Transformer**：一条路径走 BEV 全局搜索（快），另一条路径走 3D Voxel 局部精修（细）。它证明了：三维感知不一定要硬啃 3D 卷积，分层治理才是王道。

---

## 2. 🏗️ 模型架构 (Architecture Map)
![OccFormer Pipeline](https://github.com/OpenDriveLab/OccFormer/raw/main/figs/pipeline.png)
*Figure 1: OccFormer 架构。展示了双路径编码器与基于 Mask2Former 的占据解码器。*

### **详细文字描述：**
1. **Lifting**: 利用 LSS 或 Transformer 将多视角特征投影到 $128 	imes 128 	imes 16$ 的 3D 体素空间。
2. **Dual-path Encoder (核心)**:
   - **Local Path**: 在 3D Voxel 上执行 **Window-based Self-Attention**，捕获精细的几何细节。
   - **Global Path**: 将 3D Voxel 压缩为 2D BEV，执行全局自注意力，捕捉宏观场景布局。
3. **Occupancy Decoder**: 借鉴 **Mask2Former**，使用 Class-guided Query 预测各体素点的占据掩码和语义类别。

---

## 3. 💡 核心创新 (Math & Pseudo-code)

### 3.1 双路径特征聚合 (Dual-path Aggregation)
**逻辑**：局部关注 $Z$ 轴垂直一致性，全局关注 $XY$ 平面布局一致性。

**PyTorch 风格伪代码实现**：
```python
def forward_dual_path(voxel_feats):
    # voxel_feats: [B, C, Z, H, W]
    
    # 1. Global Path: BEV View
    # 将 Z 维度池化/压缩
    bev_feats = voxel_feats.mean(dim=2) 
    bev_global = self.global_transformer(bev_feats)
    
    # 2. Local Path: Windowed Voxel View
    # 在每个 [window_h, window_w] 窗口内做注意力
    voxel_local = self.local_transformer(voxel_feats)
    
    # 3. 跨层级特征融合
    # 将全局语义广播回 3D 空间，与局部细节相加
    fused_feats = voxel_local + bev_global.unsqueeze(2)
    
    return fused_feats
```

---

## 5. 📊 关键指标 (Occ3D Dataset)
- **mIoU**: 在 Occ3D 榜单上达到 **32.5%** (R50 Backbone)，在小物体（如路锥）上的召回率提升显著。
- **一致性**: 垂直方向的语义连贯性相比单层 BEV 模型提升了约 **20%**。

---

## 10. 🛠️ 落地建议
- **算力分配**: 由于 3D Local Path 计算量极大，建议仅在车辆周围 **20m** 范围内开启局部精修，远端则降级为纯 BEV 处理。
- **显存预警**: 显存占用极高，建议配合 **Activation Checkpointing** 或 **FlashAttention** 训练。
