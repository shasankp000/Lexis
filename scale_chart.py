# scale_chart.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

x_labels = ['5k', '10k', '20k', '50k', '100k']

lexis_r = {
    'char_bpb':   [2.2275, 2.3510, 2.3629, 2.3531, 2.1447],
    'full_bpb':   [10.0143, 9.3149, 8.8094, 8.2174, 7.8482],
    'overlap':    [98.25, 96.99, 95.11, 96.80, 97.18],
    'payload_kb': [6294/1024, 11795/1024, 22585/1024, 52392/1024, 99778/1024],
}
gzip = {
    'bpb':        [3.3731, 3.5917, 3.5858, 3.5030, 3.3763],
    'payload_kb': [2120/1024, 4548/1024, 9193/1024, 22334/1024, 42924/1024],
}
zstd = {
    'bpb':        [3.3206, 3.4788, 3.4473, 3.3475, 3.1656],
    'payload_kb': [2087/1024, 4405/1024, 8838/1024, 21343/1024, 40246/1024],
}
original_kb = [5028/1024, 10130/1024, 20510/1024, 51006/1024, 101708/1024]

C = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Bits per Byte — char stream",
        "Bits per Byte — full payload",
        "Payload Size (KB)",
        "Lexis-R Word Overlap %",
    ),
    vertical_spacing=0.18,
    horizontal_spacing=0.12,
)

# Chart 1: char-stream bpb
for name, data, color in [('Lexis-R (char)', lexis_r['char_bpb'], C[0]),
                           ('gzip-9',         gzip['bpb'],         C[1]),
                           ('zstd-19',        zstd['bpb'],         C[2])]:
    fig.add_trace(go.Scatter(x=x_labels, y=data, name=name, mode='lines+markers',
                             line=dict(color=color)), row=1, col=1)

# Chart 2: full payload bpb
for name, data, color in [('Lexis-R (full)', lexis_r['full_bpb'], C[0]),
                           ('gzip-9',         gzip['bpb'],         C[1]),
                           ('zstd-19',        zstd['bpb'],         C[2])]:
    fig.add_trace(go.Scatter(x=x_labels, y=data, name=name, mode='lines+markers',
                             line=dict(color=color), showlegend=False), row=1, col=2)

# Chart 3: payload size KB
for name, data, color, dash in [('Lexis-R',  lexis_r['payload_kb'], C[0], 'solid'),
                                  ('gzip-9',   gzip['payload_kb'],    C[1], 'solid'),
                                  ('zstd-19',  zstd['payload_kb'],    C[2], 'solid'),
                                  ('Original', original_kb,           C[3], 'dash')]:
    fig.add_trace(go.Scatter(x=x_labels, y=data, name=name, mode='lines+markers',
                             line=dict(color=color, dash=dash), showlegend=False), row=2, col=1)

# Chart 4: word overlap
fig.add_trace(go.Scatter(x=x_labels, y=lexis_r['overlap'], name='Word Overlap',
    mode='lines+markers', fill='tozeroy', fillcolor='rgba(99,110,250,0.12)',
    line=dict(color=C[0]), showlegend=False), row=2, col=2)

for r, c, xt, yt in [(1,1,'Input size','bpb'),(1,2,'Input size','bpb'),
                      (2,1,'Input size','Size (KB)'),(2,2,'Input size','Overlap %')]:
    fig.update_xaxes(title_text=xt, row=r, col=c)
    fig.update_yaxes(title_text=yt, row=r, col=c)

fig.update_layout(
    title=dict(text=(
        "Lexis-R vs gzip / zstd — Scale Test<br>"
        "<span style='font-size:16px;font-weight:normal;'>"
        "Moby Dick excerpts | char-stream bpb ~32% below zstd at all scales</span>"
    )),
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    template='plotly_dark',
    width=1100,
    height=820,
)

fig.write_image("lexis_r_scale_comparison.png")
print("Saved lexis_r_scale_comparison.png")
