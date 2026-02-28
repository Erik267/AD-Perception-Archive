---
date: 2026-02-27
keywords: [DriveVLM-Dual, VLM, System-1/2, CVPR 2024, Hybrid Planning]
tags: [Level-04, Perception-Hardcore, VLA-Reasoning]
---

# 2026-02-27-DriveVLM-Dual-CVPR2024-论文速读

## 0. 基本信息
- **发表时间**: 2024年 (CVPR 2024)
- **作者单位**: 清华大学、地平线、南京大学
- **代码仓库**: [https://github.com/OpenDriveLab/DriveVLM](https://github.com/OpenDriveLab/DriveVLM)
- **Tags**: #双系统推理 #VLM驾驶 #快慢思考 #认知智能

## 1. 🔪 今日锐评
> **DriveVLM-Dual** 是 VLA 工业化落地的“过渡期最优解”。
> 核心逻辑在于将 LLM 复杂的推理逻辑（System-2）作为“软先验”注入到经典的 BEV 感知栈（System-1）中。它解决了 VLM 在毫米级空间定位上的无力感，同时保留了它对复杂交通规则的深刻理解。

## 2. 🏗️ 模型架构
- **System-2 (VLM)**: 采用 Qwen-7B + InternViT，输出 CoT 推理链。
- **System-1 (Perception)**: 经典的检测、占据流模块。
- **Modulation**: VLM 的推理结果指导传统 Planner 的代价地图（Cost-map）权重。

## 3. 💡 核心创新 (Math & Pseudo-code)
### 3.1 规划引导 (Linguistic Guidance)
通过将 LLM 的语义元动作映射为 3D 空间的引导概率图，实现跨模态对齐。

```python
def modulate_cost_map(cost_map, meta_action_embed):
    # 将 LLM 元动作转化为空间引导
    guidance = self.projector(meta_action_embed)
    # 调整搜索空间
    return cost_map * torch.sigmoid(guidance)
```

## 10. 🛠️ 落地建议
- **算子合并**: 建议在 Orin 上将 Visual Projector 编译为单个 TRT 算子。
- **容错逻辑**: 必须配置 System-2 失效时的紧急 FALLBACK。
