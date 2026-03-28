# AD-Perception 全栈技术图谱 (Knowledge Graph)

## 📌 核心学习资源 (Manuals)
- **[基础必读]** [/archive/04_VLA/基础知识2.0.md](./archive/04_VLA/基础知识2.0.md)
    - **内容**: 20,000字级 VLA 终极技术手册。涵盖从视觉 Token 化、动作离散化到 ReCogDrive 源码实战的全量细节。

---

## 🟢 Level 01: 单目感知 (已完成)
- [SC-Lane (ICCV'25)](./archive/01_Monocular/2026-03-03-SC-Lane-ICCV2025-论文速读.md) - 坡度感知 3D 车道线检测。
- [MonoDETR (ICCV'23)](./archive/01_Monocular/2026-02-27-MonoDETR-ICCV2023-论文速读.md) - 深度引导 Query。
- [MonoCon (CVPR'22)](./archive/01_Monocular/2026-02-27-MonoCon-CVPR2022-论文速读.md) - 8角点几何约束。
- [BEV-LaneDet (CVPR'23)](./archive/01_Monocular/2026-02-27-BEV-LaneDet-CVPR2023-论文速读.md) - 物理对齐标杆。
- [PerspFormer (CVPR'22)](./archive/01_Monocular/2026-02-27-PerspFormer-CVPR2022-论文速读.md) - SCA 算子鼻祖。
- [TopoNet (CVPR'24)](./archive/01_Monocular/2026-02-27-TopoNet-CVPR2024-论文速读.md) - 拓扑场景图。
- [LaneGAP (ECCV'24)](./archive/01_Monocular/2026-02-27-LaneGAP-ECCV2024-论文速读.md) - 路径化连续建模。
- [MonoScene (CVPR'22)](./archive/01_Monocular/2026-02-27-MonoScene-CVPR2022-论文速读.md) - 3D 占据感知。

---

## 🔵 Level 02: 多目 BEV 与 LSS 感知 (已完成)
- [LSS (ECCV'20)](./archive/02_MultiView/2026-02-27-LSS-ECCV2020-论文速读.md) - 离散深度投影。
- [BEVFormer (ECCV'22)](./archive/02_MultiView/2026-02-27-BEVFormer-ECCV2022-论文速读.md) - 时空对齐标杆。
- [BEVDet4D (2023)](./archive/02_MultiView/2026-02-27-BEVDet4D-2023-论文速读.md) - 时序 LSS 量产方案。
- [StreamPETR (ICCV'23)](./archive/02_MultiView/2026-02-27-StreamPETR-ICCV2023-论文速读.md) - 时序 Query 传播。
- [Sparse4D (2024)](./archive/02_MultiView/2026-02-27-Sparse4D-2024-论文速读.md) - 极致稀疏化。
- [OccProphet (ICLR'25)](./archive/02_MultiView/2026-02-27-OccProphet-ICLR2025-论文速读.md) - 4D 占据预测。
- [GaussianLSS (CVPR'25)](./archive/02_MultiView/2026-02-27-GaussianLSS-CVPR2025-论文速读.md) - 3DGS 深度不确定性。
- [RenderOcc (CVPR'24)](./archive/02_MultiView/2026-02-28-RenderOcc-CVPR2024-论文速读.md) - 2D 渲染监督 3D。
- [SurroundOcc (ICCV'23)](./archive/02_MultiView/2026-02-28-SurroundOcc-ICCV2023-论文速读.md) - 3D 空间交叉注意力。
- [FB-OCC (CVPR'24)](./archive/02_MultiView/2026-02-28-FB-OCC-CVPR2024-论文速读.md) - 前向-后向双向变换。
- [OccSora (CVPR'25)](./archive/02_MultiView/2026-03-02-OccSora-CVPR2025-论文速读.md) - 4D 世界生成与占用图。
- [GaussianOcc (CVPR'25)](./archive/02_MultiView/2026-03-02-GaussianOcc-CVPR2025-论文速读.md) - 高斯泼溅高精占用图。
- [MGMapNet (ICLR'25)](./archive/02_MultiView/2025-03-10-MGMapNet-ICLR2025-论文速读.md) - 多粒度查询解决点线拓扑拉扯。
- [InteractionMap (CVPR'25)](./archive/02_MultiView/2025-03-10-InteractionMap-CVPR2025-论文速读.md) - 几何感知对齐与时空交互。
- [MapDR (CVPR'25)](./archive/02_MultiView/2025-03-10-MapDR-CVPR2025-论文速读.md) - 视觉语言驱动的规则车道关联。
- [TopoBDA (2026)](./archive/02_MultiView/2026-03-20-TopoBDA-Neurocomputing-论文速读.md) - 贝塞尔可变形注意力拓扑提取。
- [UIGenMap (CVPR'25)](./archive/02_MultiView/2025-03-10-UIGenMap-CVPR2025-论文速读.md) - 不确定性感知生成式地图构建。
---

## 🟡 Level 03: 端到端感知与规划 (已完成)
- [UniAD (CVPR'23)](./archive/03_EndToEnd/2026-02-27-UniAD-CVPR2023-论文速读.md) - 大统一先驱。
- [VAD (ICCV'23)](./archive/03_EndToEnd/2026-02-27-VAD-ICCV2023-论文速读.md) - 矢量化极速规控。
- [VAD-v2 (2024)](./archive/03_EndToEnd/2026-02-27-VAD-v2-2024-论文速读.md) - 全概率规划。
- [DriveTransformer (ICLR'25)](./archive/03_EndToEnd/2026-02-27-DriveTransformer-ICLR2025-论文速读.md) - 并行可扩展架构。
- [BridgeAD (CVPR'25)](./archive/03_EndToEnd/2026-02-28-BridgeAD-CVPR2025-论文速读.md) - 多步 Query 时序对齐。
- [Hydra-MDP (CVPR'24)](./archive/03_EndToEnd/2026-02-28-Hydra-MDP-CVPR2024-论文速读.md) - 多头蒸馏规则规划。
- [V-AD v3 (CVPR'25)](./archive/03_EndToEnd/2026-03-02-V-AD-v3-CVPR2025-论文速读.md) - 大模型驱动规划。
- [SparseAD-v2 (2025)](./archive/03_EndToEnd/2026-03-02-SparseAD-v2-2025-论文速读.md) - 极致稀疏实时规划。
- [DriveWorld (CVPR'25)](./archive/03_EndToEnd/2026-03-02-DriveWorld-CVPR2025-论文速读.md) - 时序世界模型预测。
- [GenAD (CVPR'25)](./archive/03_EndToEnd/2026-03-02-GenAD-CVPR2025-论文速读.md) - 生成式端到端驱动。
---

## 🔴 Level 04: Vision-Language-Action (VLA) (已完成)
- [Alpamayo-R1 (NeurIPS'25)](./archive/04_VLA/2026-02-27-Alpamayo-R1-NeurIPS2025-论文速读.md) - 工业级 Reasoning。
- [Impromptu VLA (NeurIPS'25)](./archive/04_VLA/2026-02-27-Impromptu-VLA-Bosch-NeurIPS2025-论文速读.md) - 数据闭环标杆。
- [DriveVLM-Dual (CVPR'24)](./archive/04_VLA/2026-02-27-DriveVLM-Dual-CVPR2024-论文速读.md) - 双系统混合推理。
- [MindDrive (2025)](./archive/04_VLA/2026-02-27-MindDrive-2025-论文速读.md) - 在线强化学习进化。
- [DriveGPT4-v2 (CVPR'25)](./archive/04_VLA/2026-02-27-DriveGPT4-v2-CVPR2025-论文速读.md) - 闭环推理性能王。
- [OmniDrive (CVPR'25)](./archive/04_VLA/2026-02-28-OmniDrive-CVPR2025-论文速读.md) - 3D 空间查询与反事实推理。
- [Reason2Drive (ECCV'24)](./archive/04_VLA/2026-02-28-Reason2Drive-ECCV2024-论文速读.md) - 思维链 CoT 逻辑推理。
- [DriveMLM (CVPR'24)](./archive/04_VLA/2026-02-28-DriveMLM-CVPR2024-论文速读.md) - 对齐行为决策状态。

---

## 🟣 Level 05: 仿真与世界模型 (已完成)
- [DIO (CVPR'25)](./archive/05_SimGen/2026-02-27-DIO-CVPR2025-论文速读.md) - 可分解 4D 世界模型。
- [Genesis (NeurIPS'25)](./archive/05_SimGen/2026-02-27-Genesis-NeurIPS2025-论文速读.md) - 物理一致多模态生成。
- [GaussianWorld (CVPR'25)](./archive/05_SimGen/2026-02-27-GaussianWorld-CVPR2025-论文速读.md) - 3DGS 流式预测。
- [DriveDreamer4D (CVPR'25)](./archive/05_SimGen/2026-02-27-DriveDreamer4D-CVPR2025-论文速读.md) - 自动驾驶数据机器。
- [Vista (NeurIPS'24)](./archive/05_SimGen/2026-02-28-Vista-NeurIPS2024-论文速读.md) - 长时序高保真世界模型。
- [MagicDrive (ICLR'24)](./archive/05_SimGen/2026-02-28-MagicDrive-ICLR2024-论文速读.md) - 3D 几何硬控街景生成。
- [TrajEdit (ICLR'25)](./archive/05_SimGen/2026-03-02-TrajEdit-ICLR2025-论文速读.md) - 闭环轨迹编辑增强。
- [DriveArena (2025)](./archive/05_SimGen/2026-03-02-DriveArena-2025-论文速读.md) - 多智能体闭环仿真平台。