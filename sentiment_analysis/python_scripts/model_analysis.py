import json
import plotly.graph_objects as go

from sentiment_analysis import DATA_DIR


with open(DATA_DIR / 'stats/my_sentiment_model_valid_e1b64_losses.json', 'r') as file:
    losses = json.load(file)

train_losses = [[e[0] * e[2] + e[1], e[3]] for e in losses['train_losses'] if e[1] % 5 == 0]
val_losses = [[e[0] * e[2] + e[1], e[3]] for e in losses['val_losses']]
train_x, train_y = zip(*train_losses)
val_x, val_y = zip(*val_losses)

layout = go.Layout(title='Losses per step', xaxis={'title': 'Step'}, yaxis={'title': 'Loss'})
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=train_x, y=train_y, name='train error', mode='lines+markers'))
fig.add_trace(go.Scatter(x=val_x, y=val_y, name='val error', mode='lines+markers'))
fig.show()
a = 1