import plotly.graph_objects as go
import networkx as nx

# 1. 核心数据
data = {
    "Monocular": ["MonoCon", "MonoDETR", "BEV-LaneDet", "PerspFormer", "TopoNet", "LaneGAP"],
    "Multi-View": ["LSS", "BEVFormer", "BEVDet4D", "StreamPETR", "Sparse4D", "OccProphet", "GaussianLSS", "MapTRv2"],
    "End-to-End": ["UniAD", "VAD", "VAD-v2", "DriveTransformer", "DiffusionDrive"],
    "VLA": ["Alpamayo-R1", "Impromptu-VLA", "DriveVLM-Dual", "MindDrive", "AutoVLA"],
    "World Model": ["DIO", "Genesis", "GaussianWorld", "DriveDreamer4D"]
}

# 演进血缘 (核心链路)
evo_edges = [
    ("PerspFormer", "BEVFormer"), ("LSS", "BEVDet4D"), ("BEVFormer", "UniAD"),
    ("UniAD", "VAD-v2"), ("LSS", "OccProphet"), ("VAD-v2", "MindDrive")
]

# 2. 计算 3D 布局
G = nx.Graph()
for hub, satellites in data.items():
    G.add_node(hub, size=14, color='white', type='hub') # 大枢纽：纯白
    for s in satellites:
        G.add_node(s, size=6, color='#BBBBBB', type='paper') # 论文点：浅灰
        G.add_edge(hub, s)
G.add_edges_from(evo_edges)

pos = nx.spring_layout(G, dim=3, k=0.4, seed=42)

# 3. 提取绘图元素
edge_x, edge_y, edge_z = [], [], []
for u, v in G.edges():
    x0, y0, z0 = pos[u]; x1, y1, z1 = pos[v]
    edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None]); edge_z.extend([z0, z1, None])

nodes_x, nodes_y, nodes_z, nodes_text, nodes_size, nodes_color = [], [], [], [], [], []
for node, d in G.nodes(data=True):
    x, y, z = pos[node]
    nodes_x.append(x); nodes_y.append(y); nodes_z.append(z)
    nodes_text.append(node)
    nodes_size.append(d['size'])
    nodes_color.append(d['color'])

# 4. 暗黑极简绘图
fig = go.Figure()

# 连边 (深灰细线)
fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', 
                           line=dict(color='#444444', width=1), hoverinfo='none'))

# 节点
fig.add_trace(go.Scatter3d(x=nodes_x, y=nodes_y, z=nodes_z, mode='markers+text', 
                           text=nodes_text,
                           marker=dict(size=nodes_size, color=nodes_color, opacity=1.0),
                           textfont=dict(color='white', size=9),
                           textposition="top center", hoverinfo='text'))

fig.update_layout(
    showlegend=False,
    paper_bgcolor='black', # 画布背景纯黑
    plot_bgcolor='black',
    scene=dict(
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False), 
        zaxis=dict(visible=False),
        bgcolor='black' # 3D 空间背景纯黑
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# 强制开启 WebGL 优化渲染
fig.write_html("AD_Relationship_Dark.html")
print("成功生成暗黑极简版：AD_Relationship_Dark.html")
