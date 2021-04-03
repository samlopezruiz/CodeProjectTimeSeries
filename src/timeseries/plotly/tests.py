import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"
if __name__ == '__main__':
    fig = go.Figure(
        data=[go.Bar(y=[2, 1, 3])],
        layout_title_text="A Figure Displayed with fig.show()"
    )
    fig.show()
    print(pio.renderers.default)