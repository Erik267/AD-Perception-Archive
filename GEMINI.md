# Gemini CLI Directive: 感知全栈极速研究流水线 (Industrial Standard)

## 1. 归档与存档规范 (Archive Norms)
- **文件路径**: `/home/lichong.i/code/paper/archive/[Level-XX]/`
- **文件命名**: `YYYY-MM-DD-PaperName-Venue-论文速读.md`
- **源码路径**: `/home/lichong.i/code/paper/repos/[Level-XX]/[Name]` (克隆或单文件审计)

## 2. 极致拆解 10 步模版 (Mandatory Template)
### 0. 基本信息
- 时间、会议/期刊、**作者单位 (Affiliation)**、官方代码仓库、专业 Tags。
### 1. 🔪 今日锐评
- 犀利的行业洞察，点出算法解决的物理痛点（如：深度抖动、因果混乱）。
### 2. 🏗️ 模型架构 (Architecture Map)
- 引用官方高清图链接 + **详尽的文字描述**（数据流、特征维度演变、算子层级）。
### 3. 💡 核心创新 (Math & Pseudo-code)
- 数学公式推导 + **PyTorch 风格伪代码**（必须标注每一行 Tensor Shape）。
### 4. 📉 Loss 函数详解
- 各分项公式、权重平衡策略、物理含义。
### 5. 📊 关键指标 (SOTA Compare)
- 具体的 Table 数据，对标 nuScenes, OpenLane, CARLA 等。
### 6. 📂 数据策略与预处理
- 数据增强逻辑、内参/外参归一化方案（如 Virtual Camera 机制）。
### 7. 🧩 时序与稳定性 (Temporal Stability)
- 时序特征融合逻辑、跳动感分析、历史记忆深度。
### 8. ⚠️ 长尾与局限 (Corner Cases)
- OOD 场景表现、遮挡弱点、算力瓶颈。
### 9. ⚖️ 优缺点总结
- 性能与精度的权衡、部署难度评分。
### 10. 🛠️ 落地建议 (Deployment)
- 算子合并策略、TensorRT 量化坑点、硬件同步（如 IMU 同步）要求。

## 3. 核心准则 (Golden Rules)
- **源码必验**: 必须校验 GitHub 链接真实性，禁止 404 链接。
- **物理优先**: 逻辑解释必须优先于参数解释。
- **2025/2026 先行**: 每周扫描必须优先捕捉最新的 SOTA 录用论文。
