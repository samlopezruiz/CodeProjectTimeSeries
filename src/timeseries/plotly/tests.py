import time
from itertools import combinations
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from timeseries.plotly.utils import plotly_params_check, plotly_save
import plotly.io as pio

pio.renderers.default = "browser"

if __name__ == '__main__':
    regimes = []
    c = 100
    for i in range(c):
        regimes.append([i, i + 1])

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.append_trace(
        go.Scatter(
            x=list(range(c)),
            y=list(range(c)),
            visible=True,
        ),
        row=1,
        col=1
    )

    colors = ['lightcyan', 'lightblue', 'dodgerblue', 'teal',
              'lightcyan', 'lightblue', 'olivedrab',  'dodgerblue', 'teal', 'orange',
              'beige', 'palegoldenrod', 'burlywood',
              'beige', 'palegoldenrod', 'burlywood', 'dodgerblue', 'teal',

              'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
               'bisque', 'black', 'blanchedalmond', 'blue',
              'blueviolet', 'brown',  'cadetblue',
              'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
              'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod',
              'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen',
              'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue',
              'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink',
              'deepskyblue', 'dimgray', 'dimgrey',  'firebrick', 'floralwhite',
              'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray',
              'grey', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
               'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon',
              'lightcoral',  'lightgoldenrodyellow', 'lightgray', 'lightgrey',
              'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue',
              'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime',
              'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue',
              'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
              'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose',
              'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange',
              'orangered', 'orchid', 'palegreen', 'paleturquoise', 'palevioletred',
              'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red',
              'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
              'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
              'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet',
              'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']

    if regimes is not None:
        for r, regime in enumerate(regimes):
            for i in range(0, len(regime), 2):
                fig.add_vrect(
                    x0=regime[i], x1=regime[i + 1],
                    fillcolor=colors[r], opacity=0.5,
                    layer="below", line_width=0,
                ),

    fig.show()
