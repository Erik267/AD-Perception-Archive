---
date: 2026-02-27
keywords: [GaussianWorld, World Model, Streaming Occupancy, 3DGS, CVPR 2025]
tags: [Level-05, World-Model, 4D-Occupancy]
---

# GaussianWorld-论文速读

## 0. 基本信息
- **发表时间**: 2025年 (CVPR 2025)
- **作者单位**: 业界领先仿真团队
- **代码仓库**: [待公开]
- **Tags**: #Gaussian世界模型 #流式占据预测 #4D场景演化 #CVPR2025

---

## 1. 🔪 今日锐评
> **GaussianWorld** 是目前把“预测”和“重建”融合得最优雅的模型。
> 
> **核心洞察**：它抛弃了传统的 Voxel Grid，改用 **3D Gaussians** 来代表整个世界。由于 Gaussian 是轻量且可运动的，它可以随着时间戳“流动”。模型不仅能重建现在的场景，还能通过预测 Gaussian 的位移来预报未来的 Occupancy。这实现了真正意义上的 **“Streaming 4D Perception”**。

---

## 3. 💡 核心创新 (Math & Pseudo-code)

### 3.1 动态 Gaussian 流动 (Streaming 4D Flow)
**逻辑**：对场景中的动态 Gaussian 预测位移矢量 $\Delta \mu$。

**PyTorch 风格伪代码实现**：
```python
def stream_world_evolution(prev_gaussians, ego_motion):
    # 1. 自车运动补偿 (Alignment)
    curr_gaussians = self.apply_ego_pose(prev_gaussians, ego_motion)
    
    # 2. 预测局部动态位移
    # 对每个 Gaussian 点预测 delta_xyz
    movement_offsets = self.flow_net(curr_gaussians.features)
    evolved_gaussians = curr_gaussians.update_pos(movement_offsets)
    
    # 3. 占据空间推导 (Inference)
    # 将演化后的高斯投影到网格，生成未来 Occupancy
    future_occ = self.rasterize_to_occupancy(evolved_gaussians)
    
    return future_occ
```

---

## 5. 📊 关键指标 (nuScenes)
- **mIoU**: 相比 Cam4DOcc 这种 Voxel 方案提升了 **2.5%**。
- **一致性**: 时空跳动（Flickering）减少了 **30%**。
