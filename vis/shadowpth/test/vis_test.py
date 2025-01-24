from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import numpy as np
import plotly.graph_objs as go

trace1 = go.Scatter(
     y = np.random.randn(700),
    mode = 'markers',
    marker = dict(
        size = 16,
        color = np.random.randn(800),
        colorscale = 'Viridis',
        showscale = True
    )
)
data = [trace1]
py.iplot(data)