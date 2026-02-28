---
date: 2026-02-27
keywords: [DiffusionDrive, Truncated Diffusion, E2E-AD, Path Planning, CVPR 2025]
tags: [Level-03, Perception-to-Planning, Diffusion-Policy]
---

# DiffusionDrive-论文速读

## 0. 基本信息
- **发表时间**: 2025年 (CVPR 2025)
- **作者单位**: 业界领先 AD 实验室
- **代码仓库**: [https://github.com/chen-yang-liu/DiffusionDrive](https://github.com/chen-yang-liu/DiffusionDrive)
- **Tags**: #截断扩散模型 #生成式规控 #实时端到端 #CVPR2025

---

## 1. 🔪 今日锐评
> **DiffusionDrive** 证明了扩散模型也能跑实时自动驾驶。
> 
> **核心洞察**：传统的扩散规划太慢了，动辄几十次迭代。DiffusionDrive 提出了 **Truncated Diffusion (截断扩散)**。它不再从纯噪声开始，而是从**多模态锚点轨迹**（Prior Anchors）出发，只做 **2 步** 去噪。这既保留了生成式模型处理复杂多意图（如：既可以绕行也可以等待）的优势，又把速度拉到了 **45 FPS**。

---

## 3. 💡 核心创新 (Math & Pseudo-code)

### 3.1 截断扩散策略 (Truncated Denoising)
**逻辑**：从 Anchor 出发，快速回归最优解。

**PyTorch 风格伪代码实现**：
```python
def forward_diffusion_plan(condition_feats, traj_anchors):
    # condition_feats: 传感器感知的上下文特征
    # traj_anchors: 预设的多种专家轨迹 [K, T, 2]
    
    # 1. 注入噪声 (少量)
    z_t = traj_anchors + torch.randn_like(traj_anchors) * 0.1
    
    # 2. 核心：2-step 快速去噪
    # 第一步：粗调
    z_1 = self.denoiser(z_t, condition_feats, step=1)
    # 第二步：精修 (Cascade Decoder)
    z_0 = self.denoiser(z_1, condition_feats, step=0)
    
    return z_0 # 最终预测的平滑轨迹
```

---

## 5. 📊 关键指标 (NAVSIM Benchmark)
- **PDMS Score**: **88.1** (刷新纪录)。
- **推理速度**: 在 Orin-X 上可稳定跑在 **20Hz** 以上。
