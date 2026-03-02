# DriveWorld: Temporal World Models for Planning-Oriented Perception

## 0. 基本信息
- **时间**: 2024/2025 (CVPR 2025)
- **作者单位**: 百度, 清华大学
- **专业 Tags**: `World Model`, `Temporal Consistency`, `Planning-Oriented`

## 1. 🔪 今日锐评
**物理痛点**: 解决了端到端规划中的 **“因果混乱（Causal Confusion）”**。模型往往会因为观察到刹车灯亮了才刹车，而不是因为预测到前方有障碍物。DriveWorld 通过“世界模型”预测场景演化，强制规划轨迹必须与预测的未来物理状态对齐。

## 2. 🏗️ 模型架构 (Architecture Map)
- **Transition Model**: 预测 $S_t \to S_{t+1}$ 的潜空间状态演化。
- **Observation Model**: 从潜空间状态恢复 3D Occupancy 场景。
- **Policy Head**: 基于多步预测的未来状态生成最优驾驶路径。

## 3. 💡 核心创新 (Math & Pseudo-code)
**潜空间动力学模型 (Latent Dynamics Model)**:
```python
# Pseudo-code for Imagination Buffer
# z: Latent state [B, 512], action: [B, 3]
def imagine_future(z, action_seq):
    future_latents = []
    curr_z = z
    for action in action_seq:
        next_z = self.transition_model(curr_z, action)
        future_latents.append(next_z)
        curr_z = next_z
    return future_latents # Shape: [B, Horizon, 512]
```

## 5. 📊 关键指标 (SOTA Compare)
- **Collision Rate**: 在 CARLA 环境下较普通 E2E 模型降低 40%。

## 10. 🛠️ 落地建议 (Deployment)
- **分布式计算**: 建议在训练阶段使用并行化的 Imagination 缓存，提升训练吞吐量。
