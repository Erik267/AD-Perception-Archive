import plotly.graph_objects as go
import networkx as nx
import re
import os

# 1. 动态解析 knowledge_graph.md
def parse_knowledge_graph(file_path):
    if not os.path.exists(file_path):
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 映射表：Level ID -> Hub Name
    level_map = {
        "Level 01": "Monocular",
        "Level 02": "Multi-View",
        "Level 03": "End-to-End",
        "Level 04": "VLA",
        "Level 05": "World Model"
    }
    
    data = {}
    # 分割各级章节
    sections = re.split(r'## .*?Level (\d+):', content)
    
    for i in range(1, len(sections), 2):
        level_num = f"Level {sections[i]}"
        hub_name = level_map.get(level_num, f"L{sections[i]}")
        body = sections[i+1]
        
        # 提取 [Paper Name]
        papers = re.findall(r'- \[(.*?)\]', body)
        # 仅保留名称，去除年份后缀以简化视觉（可选，如 "MonoCon (CVPR'22)" -> "MonoCon"）
        clean_papers = [p.split(' (')[0] for p in papers]
        data[hub_name] = clean_papers
        
    return data

# 2. 加载数据
kg_data = parse_knowledge_graph('knowledge_graph.md')

# 核心演进血缘 (硬核链路，若节点存在则连接)
evo_edges_candidates = [
    ("PerspFormer", "BEVFormer"), ("LSS", "BEVDet4D"), ("BEVFormer", "UniAD"),
    ("UniAD", "VAD-v2"), ("LSS", "OccProphet"), ("VAD-v2", "MindDrive"),
    ("VAD-v2", "V-AD v3"), ("Sparse4D", "SparseAD-v2"), ("OccProphet", "OccSora"),
    ("BEVFormer", "OccSora"), ("LSS", "GaussianLSS"), ("GaussianLSS", "GaussianOcc")
]

# 3. 计算 3D 布局
G = nx.Graph()
for hub, satellites in kg_data.items():
    G.add_node(hub, size=15, color='#222222', type='hub') # 枢纽：深黑
    for s in satellites:
        G.add_node(s, size=7, color='#777777', type='paper') # 论文：中灰
        G.add_edge(hub, s)

# 添加血缘连边
for u, v in evo_edges_candidates:
    if u in G.nodes and v in G.nodes:
        G.add_edge(u, v, weight=2)

pos = nx.spring_layout(G, dim=3, k=0.5, seed=42)

# 4. 提取绘图元素
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

# 5. 白色工业极简绘图
fig = go.Figure()

# 连边 (浅灰细线)
fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', 
                           line=dict(color='#E0E0E0', width=1.5), hoverinfo='none'))

# 节点
fig.add_trace(go.Scatter3d(x=nodes_x, y=nodes_y, z=nodes_z, mode='markers+text', 
                           text=nodes_text,
                           marker=dict(size=nodes_size, color=nodes_color, opacity=0.9),
                           textfont=dict(color='#333333', size=10),
                           textposition="top center", hoverinfo='text'))

fig.update_layout(
    showlegend=False,
    paper_bgcolor='white', # 画布背景纯白
    plot_bgcolor='white',
    scene=dict(
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False), 
        zaxis=dict(visible=False),
        bgcolor='white' # 3D 空间背景纯白
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# 写入 HTML
fig.write_html("AD_Relationship_3D.html")
print(f"成功同步 knowledge_graph.md 并生成白色版：AD_Relationship_3D.html (共 {len(G.nodes)} 个节点)")
