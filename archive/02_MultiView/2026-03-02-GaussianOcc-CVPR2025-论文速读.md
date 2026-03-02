# GaussianOcc: High-Resolution 3D Occupancy with Gaussian Splatting

## 0. 基本信息
- **时间**: 2024/2025 (CVPR 2025)
- **作者单位**: 香港中文大学 (CUHK), 华为 Noah's Ark
- **官方代码**: [https://github.com/unsuisuisui/GaussianOcc](https://github.com/unsuisuisui/GaussianOcc)
- **专业 Tags**: `3D Gaussian Splatting`, `Occupancy`, `High-Res Perception`

## 1. 🔪 今日锐评
**物理痛点**: 显存是 Occupancy 的“死穴”。GaussianOcc 彻底抛弃了“格子”概念，用 **“会呼吸的高斯点”** 来表示世界。它解决了 **边缘深度模糊**，让路缘、细电线杆的感知识别精度大幅提升。

## 2. 🏗️ 模型架构 (Architecture Map)
- **Encoder**: 常规多目 Backbone (ResNet/Swin)。
- **Gaussian Predictor**: 预测每个 2D 特征点对应的 3D 高斯参数。
- **Splatting Renderer**: 将 3D 高斯投影回 2D/3D 空间进行占用率投票。

## 3. 💡 核心创新 (Math & Pseudo-code)
**高斯占用投影算子**:
```python
def splatting_op(gaussians):
    # 利用 3DGS 渲染技术，将高斯点属性散射到 3D 体素空间
    occ_grid = scatter_mean(gaussians.opacity, gaussians.voxel_indices)
    return occ_grid
```

## 9. ⚖️ 优缺点总结
- **优点**: 显存占用仅为传统 Voxel 方案的 1/4。
- **缺点**: 动态物体的高斯点更新存在滞后性。
