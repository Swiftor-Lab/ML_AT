import plotly.graph_objects as go

def compute_witt_formula(d: int, n: int) -> int:
    def mu(m: int) -> int:
        if m == 1: return 1
        factors = []
        d_val = 2
        temp = m
        while d_val * d_val <= temp:
            if temp % d_val == 0:
                count = 0
                while temp % d_val == 0:
                    count += 1
                    temp //= d_val
                if count > 1: return 0
                factors.append(d_val)
            d_val += 1
        if temp > 1:
            factors.append(temp)
        return (-1) ** len(factors)

    if n <= 0: return 0
    total_sum = 0
    for c in range(1, n + 1):
        if n % c == 0:
            total_sum += mu(c) * (d ** (n // c))
    return total_sum // n

def generate_loop_chart(path_str: str):
    tokens = [t.strip() for t in path_str.split() if t.strip()]
    x, y = [0], [0]
    cx, cy = 0, 0
    
    movements = {
        'a': (1, 0), 'a^-1': (-1, 0),
        'b': (0, 1), 'b^-1': (0, -1)
    }
    
    for t in tokens:
        move = movements.get(t, (0, 0))
        cx += move[0]
        cy += move[1]
        x.append(cx)
        y.append(cy)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers+text', 
                             line=dict(color='#FF4B4B', width=4),
                             marker=dict(size=10, color='#1F77B4'),
                             text=[f"P{i}" for i in range(len(x))], textposition="top right"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='Start (Identity)',
                             marker=dict(color='green', size=15, symbol='diamond')))
    fig.update_layout(title="⚙️ 기하학적 루프 궤적 추적 시뮬레이션", 
                      xaxis=dict(showgrid=True, gridcolor='LightGray'),
                      yaxis=dict(showgrid=True, gridcolor='LightGray'),
                      plot_bgcolor='white', height=450)
    return fig