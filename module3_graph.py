import networkx as nx
import plotly.graph_objects as go

def generate_cayley_graph(max_depth: int = 2):
    G = nx.Graph()
    generators = ['a', 'b', 'a^-1', 'b^-1']
    inverse = {'a': 'a^-1', 'a^-1': 'a', 'b': 'b^-1', 'b^-1': 'b'}
    G.add_node("")
    
    current_layer = [""]
    for _ in range(max_depth):
        next_layer = []
        for word in current_layer:
            for gen in generators:
                if word != "":
                    if gen == inverse.get(word.split()[-1], ""): continue
                new_word = f"{word} {gen}".strip()
                G.add_edge(word, new_word)
                if new_word not in next_layer:
                    next_layer.append(new_word)
        current_layer = next_layer
    return G

def draw_cayley_graph(G):
    pos = nx.kamada_kawai_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#A0A0A0'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node if node != "" else "Identity (e)")

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text,
                            textposition="top center", hoverinfo='text',
                            marker=dict(showscale=False, color='#2CA02C', size=12, line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=0),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     plot_bgcolor='white', height=500))
    return fig