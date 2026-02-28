# 🚀 Autonomous Driving Perception & Planning Archive

Welcome to the **AD-Perception-Archive**! This repository serves as a highly structured, code-driven knowledge base and research pipeline for Autonomous Driving (AD) algorithms, maintained and driven by an industrial standard.

It focuses on dismantling the latest State-of-the-Art (SOTA) papers (primarily 2024-2026 CVPR/ICCV/ECCV/NeurIPS) through a rigorous **10-Step Ultimate Disassembly Template**, emphasizing physical logic, mathematical foundations, and pseudo-code modeling.

## 📚 Knowledge Graph Structure

The repository is organized into a progressive evolution tree spanning from foundational 3D geometry to generative world models:

- **🟢 Level 01: Monocular Perception**
  - Lane topology, 3D object detection from single images, and graph-based reasoning.
- **🔵 Level 02: MultiView BEV & Occupancy**
  - Lift-Splat-Shoot (LSS) view transformations, 3D Occupancy Networks, and memory-efficient spatial representations.
- **🟡 Level 03: End-to-End Autonomous Driving**
  - UniAD architectures, vectorized planning, and temporal alignment mechanisms (e.g., multi-step query).
- **🔴 Level 04: Vision-Language-Action (VLA)**
  - Embodied AI for driving, multimodal reasoning, chain-of-thought (CoT), and behavioral decision states.
- **🟣 Level 05: Simulation & World Models**
  - Action-conditioned video generation, 4D controllable street views, and closed-loop data engines.

## 🛠️ The 10-Step Disassembly Protocol
Every paper archived here undergoes a strict dissection process:
1. **Basic Info**: Venue, Affiliation, Official Code.
2. **Sharp Commentary**: Identifying the core physical pain points resolved.
3. **Architecture Map**: Data flow and operator hierarchies.
4. **Core Innovation**: Math derivation + **Tensor-Shape Annotated PyTorch Pseudo-code**.
5. **Loss Function Detail**: Formulation and balancing strategies.
6. **Key Metrics**: NuScenes/OpenLane/CARLA benchmarks.
7. **Data Strategy**: Augmentation and virtual camera normalization.
8. **Temporal Stability**: History memory and jitter analysis.
9. **Corner Cases**: Limitations and OOD performance.
10. **Deployment Advice**: Operator fusion, TensorRT quantization, and hardware sync.

## 🔗 Quick Navigation
- Check out the full [Knowledge Graph](./knowledge_graph.md) for a complete list of archived papers.
- Dive into the [Archive Folder](./archive/) to read the detailed Markdown reports.

---
*Driven by extreme speed and high-fidelity code auditing.*
