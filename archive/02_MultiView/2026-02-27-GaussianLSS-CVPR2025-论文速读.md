---
date: 2026-02-27
keywords: [GaussianLSS, 3D Gaussian Splatting, Depth Uncertainty, BEV Perception, CVPR 2025]
tags: [Level-02, Perception-Hardcore, Gaussian-Splatting]
---

# GaussianLSS-论文速读

## 0. 基本信息
- **发表时间**: 2025年 (CVPR 2025)
- **作者单位**: 浙江大学、地平线
- **代码仓库**: [https://github.com/hustvl/GaussianLSS](https://github.com/hustvl/GaussianLSS)
- **Tags**: #3DGS #深度不确定性 #BEV感知 #CVPR2025

---

## 1. 🔪 今日锐评
> **GaussianLSS** 终于把 3DGS 带进了 BEV 感知的核心圈。
> 
> **核心洞察**：传统的 LSS 预测的是死板的深度 Bin，遇到遮挡或反光就容易“幻觉”。GaussianLSS 引入了连续的 **Gaussian 分布** 来描述深度。它不只是预测深度在哪，还预测了深度分布的**置信度（Uncertainty）**。这种“软采样”让 BEV 特征图在物体边缘变得极其锐利，彻底解决了远端物体定位漂移的顽疾。

---

## 3. 💡 核心创新 (Math & Pseudo-code)

### 3.1 不确定性感知采样 (Uncertainty-Aware Range)
**物理逻辑**：利用 Gaussian 均值 $\mu$ 和方差 $\sigma$ 确定采样区间。
**公式**：
$$	ext{Sampling Range} = [\mu - k\sigma, \mu + k\sigma]$$
- $k$: 容错系数。方差越小（越自信），采样越集中；方差越大，采样越发散。

**PyTorch 风格伪代码实现**：
```python
def forward_gaussian_lss(image_feats):
    # 1. 预测深度 Gaussian 参数 [B*N, 2, H, W]
    # mu: 深度均值, sigma: 深度标准差
    depth_params = self.depth_net(image_feats)
    mu, sigma = depth_params[:, 0], depth_params[:, 1]
    
    # 2. 构造 3D Gaussians
    # 每个像素对应一个沿视线方向伸缩的 Gaussian 椭球
    gaussians = self.create_gaussians(mu, sigma, image_feats)
    
    # 3. 快速高斯渲染 (Gaussian Splatting)
    # 将 3D Gaussians 投影并光栅化到 BEV 平面
    bev_feats = self.gs_renderer(gaussians)
    
    return bev_feats
```

---

## 5. 📊 关键指标 (nuScenes)
- **mAP**: 相比传统 LSS (BEVDet) 提升了 **3.2%**。
- **定位误差 (ATE)**: 在 50m 以外的区域误差降低了 **18%**。
