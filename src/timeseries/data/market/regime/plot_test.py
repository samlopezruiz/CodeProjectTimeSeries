import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"

if __name__ == '__main__':
    y1 = np.random.randn(200) - 1
    y2 = np.random.randn(200)
    y3 = np.random.randn(200) + 1
    x = np.linspace(0, 1, 200)

    colors = ['#3f3f3f', '#00bfff', '#ff7f00']

    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[1., 1., 1.],
        specs=[[{"type": "scatter"}, {"type": "xy"}],
               [{"type": "scatter"}, {"type": "xy", "rowspan": 2}],
               [{"type": "scatter"},            None           ]])

    fig.add_trace(
        go.Scatter(x = x,
                    y = y1,
                    hoverinfo = 'x+y',
                    mode='lines',
                    line=dict(color='#3f3f3f',
                    width=1),
                    showlegend=False,
                    ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x = x,
                    y = y2,
                    hoverinfo = 'x+y',
                    mode='lines',
                    line=dict(color='#00bfff',
                    width=1),
                    showlegend=False,
                    ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x = x,
                    y = y3,
                    hoverinfo = 'x+y',
                    mode='lines',
                    line=dict(color='#ff7f00',
                    width=1),
                    showlegend=False,
                    ),
        row=3, col=1
    )

    boxfig= go.Figure(data=[go.Box(x=y1, showlegend=False, notched=True, marker_color="#3f3f3f", name='3'),
                            go.Box(x=y2, showlegend=False, notched=True, marker_color="#00bfff", name='2'),
                            go.Box(x=y3, showlegend=False, notched=True, marker_color="#ff7f00", name='1')])

    for k in range(len(boxfig.data)):
         fig.add_trace(boxfig.data[k], row=1, col=2)

    group_labels = ['Group 1', 'Group 2', 'Group 3']
    hist_data = [y1, y2, y3]

    distplfig = ff.create_distplot(hist_data, group_labels, colors=colors,
                             bin_size=.2, show_rug=False)

    for k in range(len(distplfig.data_map)):
        fig.add_trace(distplfig.data_map[k],
                      row=2, col=2
                      )
    fig.update_layout(barmode='overlay')
    fig.show()