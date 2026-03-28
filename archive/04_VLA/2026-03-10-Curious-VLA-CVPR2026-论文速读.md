# Curious-VLA: Adaptive Exploration for Vision-Language-Action Models (CVPR 2026)

### 0. 基本信息
- **时间**: 2026-03 (CVPR 2026 Accepted)
- **会议**: CVPR 2026
- **作者单位 (Affiliation)**: 北京航空航天大学 (Beihang)、清华大学 (THU-AIR)、联想研究院
- **官方代码**: [Mashiroln/curious_vla](https://github.com/Mashiroln/curious_vla)
- **专业 Tags**: `End-to-End AD`, `VLA`, `Reinforcement Learning`, `Exploration`, `Narrow Policy`

### 1. 🔪 今日锐评
**物理痛点直击：端到端模型的“窄策略 (Narrow Policy)”**
当前的 VLA 模型过度依赖模仿学习 (Imitation Learning)，导致它们变成了只会开“顺风车”的乖乖女。模型高度拟合人类专家的单一轨迹，遇到分布外 (OOD) 场景（如前车突然加塞、道路临时施工）时，缺乏主动探索不同物理可行解的能力，往往直接输出极端的死锁动作。
**Curious-VLA 的降维打击**：赋予模型“好奇心”与“试探精神”。它在模仿学习中引入了**可行轨迹扩展 (FTE)**，并在强化学习阶段设计了**自适应多样性感知采样 (ADAS)**，鼓励模型在安全边界内去探索那些“高价值但罕见”的动作，彻底拓宽了智驾大脑的决策解空间。

### 2. 🏗️ 模型架构 (Architecture Map)
Curious-VLA 建立在轻量化且强大的多模态底座之上，采用了两阶段演进架构：
- **Backbone**: Qwen2.5-VL (3B) 作为核心 VLM。
- **输入空间**: 多视角图像 (Multi-view Images) + 文本指令 (Text Instructions) + 车辆自身状态 (Ego-status)。
- **第一阶段 - 探索式模仿学习 (Exploratory IL)**: 
  通过 FTE 模块，基于动力学约束将单一的 Expert Trajectory 扩增为一个“扇形的合规轨迹簇”，让模型学会“条条大路通罗马”。
- **第二阶段 - 多样性强化学习 (Diversity-aware RL)**:
  利用 ADAS 策略进行采样，并采用 SDR (Spanning Driving Reward) 奖励函数进行 PPO/GRPO 更新。

### 3. 💡 核心创新 (Math & Pseudo-code)
核心在于**自适应多样性感知采样 (Adaptive Diversity-Aware Sampling, ADAS)**。在 RL 探索中，不仅看重 Reward 高不高，还看重这个动作是否足够“新颖”（多样性高）。

**PyTorch 风格伪代码**:
```python
import torch
import torch.nn.functional as F

def adas_sampling(trajectories, rewards, base_policy_probs, lambda_div=0.5, temp=1.0):
    """
    trajectories: [B, N_samples, Horizon, 2]
    rewards: [B, N_samples] (SDR 奖励分数)
    base_policy_probs: [B, N_samples] (基础策略给出的原始概率)
    """
    B, N = rewards.shape
    
    # 1. 计算多样性得分 (Diversity Score)
    # 物理直觉：当前轨迹与专家基准轨迹（或均值轨迹）的欧式距离差异
    mean_traj = trajectories.mean(dim=1, keepdim=True) # [B, 1, Horizon, 2]
    diversity_scores = torch.norm(trajectories - mean_traj, dim=-1).mean(dim=-1) # [B, N]
    
    # 2. 融合奖励与多样性 (Exploration + Exploitation)
    # SDR (Spanning Driving Reward) 确保了物理安全性
    combined_logits = (rewards / temp) + lambda_div * diversity_scores
    
    # 3. 计算采样权重并采样 (Multinomial Sampling)
    sampling_weights = F.softmax(combined_logits, dim=-1) # [B, N]
    
    # 选出 K 个最高价值且最具探索性的样本进行 RL 更新
    K = 4
    selected_indices = torch.multinomial(sampling_weights, num_samples=K, replacement=False)
    
    return trajectories[torch.arange(B).unsqueeze(1), selected_indices]
```

### 4. 📉 Loss 函数详解
$L_{total} = L_{IL} + \alpha L_{RL} + \beta L_{reg}$

- **$L_{IL}$ (分布对齐损失)**: 在 FTE 扩增出的轨迹簇上计算 KLD (KL 散度) 或 Cross-Entropy，使模型的预测分布能够覆盖多条合规路线，而非坍缩到一点。
- **$L_{RL}$ (GRPO 策略梯度)**: 基于 SDR 奖励函数优化采样策略。SDR 会对碰撞和偏离车道施加极强的负惩罚，同时对平滑的高多样性变道施加正向激励。
- **$L_{reg}$ (动力学正则化)**: 限制轨迹的曲率 (Jerk) 和加速度分布，防止探索出物理上无法执行（翻车）的轨迹。

### 5. 📊 关键指标 (SOTA Compare)
在 **NAVSIM** (基于 nuPlan 的闭环基准) 上表现优异，显著提升了高难度博弈场景的通过率：
- **PDMS (Overall Score)**: **90.3** (超越 AutoVLA 的 89.1)。
- **Best-of-N (N=6)**: **94.8** (接近人类专家上限，展现了探索出最优解的强大潜力)。
- **TTC (Time to Collision)**: 相比 Baseline 提升 15%，意味着模型学会了提前进行防御性避让。

### 6. 📂 数据策略与预处理
- **FTE (Feasible Trajectory Expansion)**: 这是数据处理的核心。在 nuPlan 离线数据中，利用自行车运动学模型 (Bicycle Model) 对真实轨迹加入横向和纵向的微小扰动，过滤掉碰撞轨迹后，形成 Dense Trajectory Tree 作为训练目标。
- **Action Tokenization**: 将二维连续轨迹点量化为 1024 个离散词表，以便 Qwen2.5-VL 像输出文本一样自回归输出动作。

### 7. 🧩 时序与稳定性 (Temporal Stability)
- **Step-wise Normalization**: 针对长时序自回归预测容易产生误差累积（漂移）的问题，在每个预测时间步引入归一化操作，将轨迹点重新锚定到自车坐标系。
- **KV Cache 时序推理**: 利用 Transformer 的 KV Cache 机制，保留过去 2 秒的历史视觉 Token 缓存，极大降低了时序融合的计算延迟。

### 8. ⚠️ 长尾与局限 (Corner Cases)
- **安全边界的脆弱性**: RL 探索的底线在于 Reward 函数的准确性。如果仿真器的碰撞检测存在 Bug 或延迟，ADAS 采样可能会让模型“学坏”，探索出危险边缘的试探动作。
- **算力开销**: 无论是 FTE 预处理还是在线 RL 多样性采样，都需要维护多条候选轨迹的梯度图，对显存 (VRAM) 消耗极大。

### 9. ⚖️ 优缺点总结
- **优点**: 直击端到端“不敢开、死板”的痛点，显著增强了 VLA 模型在复杂开放道路上的博弈和绕行能力。
- **缺点**: 训练管线复杂，强化学习的收敛速度和稳定性高度依赖超参数 (如 `lambda_div` 和 `temp`) 的精细调节。

### 10. 🛠️ 落地建议 (Deployment)
- **量化与算力分配**: 模型部署时建议使用 **W4A8 量化** 以塞入单张 Orin-X 芯片。同时，强烈建议利用 NPU 加速 KV Cache 读写，保证多视角图像的特征不被频繁重算。
- **确定性退化兜底**: 在车端推理时，若感知到极高风险场景（如高速前车急刹），应通过调节 Temperature $T \rightarrow 0$ 强行关闭“探索性”，回退到最保守的贪心解码（Greedy Decoding）策略。