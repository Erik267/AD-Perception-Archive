# 2026-02-28-SurroundOcc-ICCV2023-论文速读

### 0. 基本信息
- **时间**: 2023 (ICCV 2023)
- **作者单位**: 清华大学 (Tsinghua University)、天津大学、PhiGent Robotics
- **官方代码仓库**: [weiyithu/SurroundOcc](https://github.com/weiyithu/SurroundOcc)
- **专业 Tags**: `3D Occupancy Prediction`, `Multi-Camera Perception`, `Scene Reconstruction`, `Autonomous Driving`

### 1. 🔪 今日锐评
SurroundOcc 是 3D 占用预测（Occupancy Prediction）领域的开山之作之一。它犀利地指出了传统 BEV 感知在处理**垂直空间结构**（如立交桥、悬空物体）时的物理局限性，并摒弃了将 3D 空间强行压缩为 2D BEV 的做法。其核心贡献在于通过 **2D-3D 空间交叉注意力**直接构建 3D 体素特征，并提出了一套无需人工标注、基于泊松重建（Poisson Reconstruction）的**稠密真值生成流水线**，解决了 Occupancy 任务长期缺乏高质量训练数据的痛点。

### 2. 🏗️ 模型架构 (Architecture Map)
SurroundOcc 采用了一种对称的 2D-3D U-Net 结构：
1.  **2D 特征提取**: 使用 ResNet-101 或 EfficientNet-B7 作为 Backbone，配合 FPN 提取多尺度（1/8, 1/16, 1/32）的图像特征。
2.  **2D-3D 空间提升 (Lifting)**: 核心算子是 **Spatial Cross-Attention**。对于 3D 空间中的每个 Query 点，根据相机内外参投影回 2D 图像，利用可变形注意力（Deformable Attention）采样特征。
3.  **3D 特征融合**: 得到多尺度 3D 卷特征（Volume Features）后，通过 3D 卷积进行特征增强，并使用 3D 反卷积（Deconv）逐层上采样，实现从低分辨率到高分辨率的特征融合。
4.  **多尺度预测**: 在 3D U-Net 的每个层级输出 Occupancy 预测，实施深层监督。

### 3. 💡 核心创新 (Math & Pseudo-code)
**核心公式：2D-3D 空间交叉注意力**
对于 3D 查询点 $Q_p$，其特征更新逻辑为：
$$F_p = \sum_{i \in V_{hit}} 	ext{DeformAttn}(Q_p, \mathcal{P}(q_p, i), X_i)$$
其中 $\mathcal{P}$ 是 3D 到 2D 的投影函数，$V_{hit}$ 是该点投影落入的相机视图集合。

**PyTorch 风格伪代码 (Tensor Shape 标注):**
```python
# B: Batch, N: Cam Num, C: Dim, H, W: Img Size, X, Y, Z: Voxel Size
def forward(self, images):
    # 1. 2D Backbone + FPN
    # feats_2d: List of [B*N, C, H/s, W/s]
    feats_2d = self.backbone_2d(images) 
    
    # 2. 2D-3D Spatial Cross Attention (Multi-scale)
    # queries_3d: [B, C, X/s, Y/s, Z/s]
    volume_feats = []
    for scale in [32, 16, 8]:
        feat_3d = self.spatial_attention(queries_3d[scale], feats_2d[scale], calibs)
        volume_feats.append(feat_3d)
        
    # 3. 3D U-Net Upsampling
    # out_3d: [B, C, X/8, Y/8, Z/8] -> [B, C, X/4, Y/4, Z/4]
    curr_feat = volume_feats[0]
    preds = []
    for i in range(1, len(volume_feats)):
        curr_feat = self.deconv3d(curr_feat) + volume_feats[i]
        preds.append(self.classifier(curr_feat)) # [B, Class, X_s, Y_s, Z_s]
        
    return preds
```

### 4. 📉 Loss 函数详解
采用多尺度深层监督，总 Loss 为各层级加权和：
$$L = \sum_{j=1}^M \alpha_j (L_{ce}^j + L_{sem-aff}^j)$$
-   **$L_{ce}$ (Cross-Entropy)**: 多分类交叉熵，用于监督每个体素的语义类别。
-   **$L_{sem-aff}$ (Scene-Class Affinity)**: 引入自 MonoScene，通过计算体素间的语义亲和力，增强局部几何一致性。
-   **$\alpha_j$ (Decayed Weight)**: 权重随分辨率增加而衰减，通常设为 $\alpha_j = 1/2^j$，优先保证高分辨率预测的准确性。

### 5. 📊 关键指标 (SOTA Compare)
在 nuScenes 验证集上（使用其生成的稠密 GT）：
-   **mIoU**: 相比 TPVFormer 和 MonoScene 有显著提升。
-   **几何重建**: 在不带语义的场景重建任务中，IoU 表现优异。
-   **nuScenes Occupancy**: 在早期榜单中处于领先地位。

### 6. 📂 数据策略与预处理
**稠密真值生成 (GT Generation Pipeline):**
1.  **多帧融合**: 将动态物体（基于 BBox 提取）和静态场景的多帧 LiDAR 点云分别对齐融合。
2.  **泊松重建 (Poisson Reconstruction)**: 对融合后的稀疏点云进行曲面重建，填补激光雷达扫描的空洞。
3.  **体素化**: 对重建后的 Mesh 进行体素化采样，生成 $200 	imes 200 	imes 16$ 或更高分辨率的稠密标签。

### 7. 🧩 时序与稳定性 (Temporal Stability)
SurroundOcc 原生版本主要关注空间融合，时序特征主要体现在其 **GT 生成阶段**（利用了历史帧点云）。在模型推理阶段，它主要依赖单帧多相机图像。后续改进版本通常会引入时序 Query 融合（类似 BEVDet4D）来增强运动物体的预测稳定性。

### 8. ⚠️ 长尾与局限 (Corner Cases)
-   **远距离感知**: 受限于相机分辨率和投影几何，远距离（>50m）体素的特征采样非常稀疏，预测精度下降明显。
-   **计算开销**: 3D 卷积和高分辨率体素 Query 带来了巨大的显存压力。
-   **遮挡弱点**: 对于完全遮挡区域，模型倾向于根据上下文“脑补”，在复杂城市路口可能出现幻觉。

### 9. ⚖️ 优缺点总结
-   **优点**: 真正的 3D 空间感知，不丢失高度信息；强大的稠密 GT 生成能力。
-   **缺点**: 显存占用极高，推理速度较慢（非实时）；缺乏显式的深度监督，完全依赖 Attention 学习几何。

### 10. 🛠️ 落地建议 (Deployment)
-   **算子合并**: 2D-3D 投影采样算子需定制 CUDA Kernel 或使用 TensorRT 的插件优化。
-   **量化坑点**: 3D 卷积在 INT8 量化时容易出现精度崩塌，建议对关键的 Attention 模块保留 FP16。
-   **硬件同步**: 极度依赖相机内外参的准确性，需确保多相机触发同步在毫秒级误差内。