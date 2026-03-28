# InteractionMap: Improving Online Vectorized HDMap Construction with Interaction (CVPR 2025)

### 0. 基本信息
- **时间**: CVPR 2025 (Accepted)
- **会议**: CVPR 2025
- **作者单位 (Affiliation)**: 华中科技大学 (HUST)、地平线 (Horizon Robotics)
- **官方代码**: [GitHub (占位待放)](https://github.com/ycscience86/InteractionMap)
- **专业 Tags**: `Vectorized HD Map`, `Geometry-Aware Alignment`, `Temporal Fusion`, `Position Relation Prior`

### 1. 🔪 今日锐评
**物理痛点直击：感知与规控的“盲目自信”**
在目前的端到端矢量化建图任务（如 MapTR）中，常常出现一个让规控团队抓狂的问题：“分类-回归失配”。模型可能给一条被严重遮挡、画得像波浪线一样的车道线，打出 $0.99$ 的极高分类置信度。规控系统信以为真，直接导致方向盘乱打。这种“高分低能”源于模型在训练时，分类 Loss 和坐标回归 Loss 是各算各的，缺乏物理层面的绑定。
**InteractionMap 的降维打击**：提出**几何感知对齐 (Geometry-aware Alignment)** 模块。它强行在底层将“预测线段与真实线段的物理重叠度 (IoU)”作为分类置信度的打分依据。你的线画得越歪，你的分类得分就必须越低。同时辅以多层级的时空显式交互，彻底治好了模型的“盲目自信”。

### 2. 🏗️ 模型架构 (Architecture Map)
InteractionMap 延续了 DETR 范式，重点重构了 Decoder 内部的交互机制：
1. **BEV Encoder**: 多视角图像 $\rightarrow$ BEV 特征（支持 BEVFormer 或 LSS 架构）。
2. **Hierarchical Temporal Fusion**: 层次化时序融合，不仅传上一帧的 Query，还通过关键帧机制提取低频全局背景，抵抗大车遮挡导致的短时致盲。
3. **Relation Decoder (核心)**:
   - **显式位置关系先验 (Position Relation Prior)**: 不再让 Query 盲目 Self-Attention，而是预先计算点与点、实例与实例的几何距离，作为 Attention 的 Mask 和 Weight。
4. **Geometry-aware Alignment Head**: 取代了传统的分类头，联合输出置信度。

### 3. 💡 核心创新 (Math & Pseudo-code)
**核心逻辑：几何感知分类得分校准**
不再孤立预测 Class，而是预测 $P_{aligned} = P_{cls} \times (\text{IoU}_{vector})^\alpha$。

**PyTorch 风格伪代码 (Geometry-aware Alignment)**:
```python
import torch
import torch.nn as nn

class GeometryAwareHead(nn.Module):
    def __init__(self, embed_dim, num_classes, alpha=1.0):
        super().__init__()
        self.cls_branch = nn.Linear(embed_dim, num_classes)
        self.reg_branch = MLP(embed_dim, out_dim=N_points * 2)
        self.alpha = alpha

    def forward(self, query_feat, gt_pts=None):
        # query_feat: [Batch, N_queries, Dim]
        
        # 1. 预测原始分类 Logits 和 坐标点
        raw_cls_logits = self.cls_branch(query_feat) # [B, N, Cls]
        pred_pts = self.reg_branch(query_feat).view(B, N, N_points, 2)
        
        if self.training and gt_pts is not None:
            # 2. 训练阶段：计算预测矢量与真值的几何重叠度 (IoU)
            # 使用类似于 Chamfer Distance 转换的 Vector IoU
            iou_score = compute_vector_iou(pred_pts, gt_pts) # [B, N]
            iou_score = torch.clamp(iou_score, min=1e-6, max=1.0)
            
            # 3. 计算对齐目标 (Soft Label)
            raw_cls_probs = torch.sigmoid(raw_cls_logits)
            
            # 物理直觉：如果 IoU 很低，即使原始分类准确，也强制压低其 Target
            aligned_target = raw_cls_probs.detach() * (iou_score.unsqueeze(-1) ** self.alpha)
            
            # 计算几何感知损失 (可采用 Varifocal Loss)
            loss_g_cls = varifocal_loss(raw_cls_logits, aligned_target)
            return raw_cls_logits, pred_pts, loss_g_cls
        
        return raw_cls_logits, pred_pts
```

### 4. 📉 Loss 函数详解
总损失函数：$L = L_{pts} + \lambda_1 L_{g-cls} + \lambda_2 L_{dir}$
- **$L_{pts}$**: L1 坐标回归损失。
- **$L_{g-cls}$ (Varifocal Loss)**: 这是其精髓。针对正样本，回归目标不再是硬标签 `1`，而是该预测框的矢量 IoU 分数；针对负样本，回归目标是 `0`。
- **$L_{dir}$ (方向一致性)**: 约束相邻控制点的切线方向，解决“线段折线化”问题。

### 5. 📊 关键指标 (SOTA Compare)
在 **nuScenes** 和 **Argoverse2** 上均展现出极强统治力：
- **nuScenes (ResNet-50)**: 相比 MapTRv2，mAP 提升约 **2.5-3.0** 个点，特别是在形态极度不规则的 Pedestrian Crossing（人行横道）上提升了超过 4 个点。
- **时序跳动率 (Jittering Rate)**: 得益于层次化时序和几何对齐，连续帧之间的车道线抖动幅度降低了 **15%**，这对车端规控（Planning）是巨大的福音。

### 6. 📂 数据策略与预处理
- **矢量 IoU 计算机制**: 传统 2D 框有确定的面积可以算 IoU，但一条“线”没有面积。InteractionMap 采用了一种将线段膨胀（Dilation）为带有宽度的多边形（Polygon），然后再求交并比的近似算法。

### 7. 🧩 时序与稳定性 (Temporal Stability)
- 独创 **Hierarchical Temporal Fusion**: 
  - **Local Fusion**: 取前 $T-1$ 帧的 Query，保持短期连贯性。
  - **Global Fusion**: 抽取历史一段时间内的关键帧（Key-frames），提炼出一个全局上下文 Token，防止模型在过路口时被局部车辆短暂遮挡而彻底“失忆”。

### 8. ⚠️ 长尾与局限 (Corner Cases)
- **极度密集的车道线**: 在如收费站等极度密集且平行的车道线场景，由于 Query 之间的位置关系先验极为相似，依然会出现匹配混乱（Bipartite Matching 震荡）。
- **计算开销**: 显式的点级和实例级位置关系图（Relation Map）构建在遇到大量 Query 时（如 $N=200$），矩阵乘法的开销会显著增加。

### 9. ⚖️ 优缺点总结
- **优点**: 极大地提高了模型输出的“可信度”，高置信度即代表高物理精度，为下游规划模块排除了巨雷。
- **缺点**: 为了计算几何先验和时序融合，整体推理延迟比极致轻量化模型（如 MapTR-nano）略高。

### 10. 🛠️ 落地建议 (Deployment)
- **动态 Query 裁剪 (Dynamic Query Pruning)**: 在部署时，建议在上层 Attention 后，根据置信度提早过滤掉一半的死 Query，以降低后续复杂 Relation 计算的维度。
- **FP16 兼容性**: 几何对齐模块中的指数运算（`iou ** alpha`）在某些 NPU/DSP 上可能精度溢出，建议将其转化为查表法（Lookup Table, LUT）或固定精度算子。