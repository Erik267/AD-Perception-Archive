---
date: 2026-02-27
keywords: [StreamPETR, Temporal Query Propagation, 3D Position Embedding, ICCV 2023]
tags: [Level-02, Perception-Hardcore, Efficiency-SOTA]
---

# StreamPETR-论文速读

## 0. 基本信息
- **发表时间**: 2023年 (ICCV 2023)
- **作者单位**: 华中科技大学、地平线 (Horizon Robotics)
- **代码仓库**: [https://github.com/exiawsh/StreamPETR](https://github.com/exiawsh/StreamPETR)
- **Tags**: #稀疏感知 #时序Query传播 #高性价比 #nuScenes-SOTA

---

## 1. 🔪 今日锐评
> **StreamPETR** 彻底终结了“一定要有 BEV 特征图”的执念。
> 
> **核心洞察**：它证明了感知不需要昂贵的 BEV 空间转换，只需要把 **3D Position Embedding (3D PE)** 玩明白。通过 **Object-Centric** 的时序传播，它让 Query 像河流一样在帧间流动，不仅极其省显存，还能白嫖长时序的运动语义。这是目前在 Orin-X 上跑 3D 检测最硬核的架构之一。

---

## 2. 🏗️ 模型架构 (Architecture Map)
![StreamPETR Pipeline](https://github.com/exiawsh/StreamPETR/raw/main/figs/streampetr_framework.png)
*Figure 1: StreamPETR 架构。展示了 Memory Queue 如何在帧间传递 Object Queries。*

### **详细文字描述：**
1. **Feature Extraction**: 标准 2D Backbone 提取多视角图像特征。
2. **3D PE Generator**: 将相机内外参编码为 3D 空间位置嵌入，与 2D 特征融合。
3. **Temporal Propagation (核心)**: 
   - **Memory Queue**: 存储上一帧的高置信度 Query。
   - **Motion Compensation**: 利用自车运动 (Ego-motion) 补偿 3D PE 的位置偏移。
4. **Transformer Decoder**: 当前帧 Query 与历史 Query 拼接后进入 Decoder，通过 Self-Attention 实现时序交互。

---

## 3. 💡 核心创新 (Math & Pseudo-code)

### 3.1 时序 Query 传播 (Temporal Propagation)
**物理逻辑**：物体在 3D 空间是连续运动的，Query 的位置也应随之演化。

**PyTorch 风格伪代码实现**：
```python
def forward_temporal(curr_queries, memory_queue, ego_motion):
    # 1. 历史 Query 对齐 (Motion-Aware)
    # prev_queries: [N, C]
    prev_queries = memory_queue.pop()
    # 根据自车运动更新 Query 的位置编码 (3D PE)
    prev_queries_aligned = apply_ego_motion(prev_queries, ego_motion)
    
    # 2. 时序融合 (Temporal Interaction)
    # 拼接当前与历史 Query
    # combined_queries: [N_curr + N_hist, C]
    all_queries = torch.cat([curr_queries, prev_queries_aligned], dim=0)
    
    # 3. 通过 Self-Attention 让当前 Query 继承历史物体的 ID 和速度
    updated_queries = self.self_attn(all_queries)
    
    return updated_queries[:N_curr]
```

---

## 4. 📉 Loss 函数详解
$$L = L_{cls} + \lambda L_{reg} + \lambda_{vel} L_{velo}$$
- **$L_{vel}$ (关键)**：通过时序 Query 差值直接回归物体速度，不依赖复杂的后处理。

---

## 5. 📊 关键指标 (nuScenes Test)
| 模型 | NDS ↑ | mAP ↑ | 延迟 (A100) |
| :--- | :--- | :--- | :--- |
| BEVFormer | 56.9 | 48.1 | 130ms |
| **StreamPETR** | **67.6 (+10.7)** | **55.0 (+6.9)** | **32ms (4x faster)** |

---

## 10. 🛠️ 落地建议
- **显存控制**: 相比 BEVFormer，StreamPETR 节省了约 60% 的显存，非常适合在 8G/16G 显存的嵌入式平台上跑大分辨率输入。
- **冷启动**: 第一帧由于 Memory Queue 为空，预测精度会略低，建议在初始化阶段增加 Dummy 帧。
