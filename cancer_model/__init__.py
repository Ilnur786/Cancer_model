import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from math import log, exp
import pickle
from torch.nn import Sequential, ReLU, Linear, Softmax
from torch.nn.init import kaiming_normal_, zeros_
from torch import load, Tensor
import pkg_resources

# mms_path = pkg_resources.resource_stream(__name__, 'cancer_model/mms.pickle')

# with open(mms_path, 'rb') as life:
#     mms = pickle.load(life)
#
# with pkg_resources.resource_stream(__name__, ".tld_set_snapshot") as snapshot_file:
#     self._extractor = _PublicSuffixListTLDExtractor(pickle.load(snapshot_file))
#     return self._extractor

with pkg_resources.resource_stream(__name__, 'mms.pickle') as mms_path: # (название пакета, название файла в пакете)
    mms = pickle.load(mms_path)

l1 = Linear(3, 50)
l2 = Linear(50, 50)
l3 = Linear(50, 2)
kaiming_normal_(l1.weight)
kaiming_normal_(l2.weight)
kaiming_normal_(l3.weight)
zeros_(l1.bias)
zeros_(l2.bias)
zeros_(l3.bias)

mlpc = Sequential(l1, ReLU(), l2, ReLU(), l3, Softmax(1))

# mlpc.load_state_dict(load('cancer_model/model_state_dict.pt'))

mlpc.load_state_dict(load(pkg_resources.resource_stream(__name__, 'model_state_dict.pt')))
mlpc.eval()



css = [{'rel': "stylesheet",
        'href': "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"}]

app = dash.Dash(__name__, external_stylesheets=css)  # мое веб-приложение

colors = {
    'background': '#D6CBE0',
    'text': '#000000'
}

inputs = [html.Label(['AGE', html.Div(dcc.Input(id='AGE', type='number', value='18', min=10, max=65))],
                     className='col-md-12'),
          html.Label(['CA125', html.Div(dcc.Input(id='CA125', type='number', min=0, max=10000))],
                     className='col-md-12'),
          html.Label(['HE4', html.Div(dcc.Input(id='HE4', type='number', min=0, max=10000)),
                      html.Hr(),
                      html.Div(html.Button('Predict', id='button', className="btn btn-dark"), className='col-md-12')],
                     className='col-md-12')]

row = html.Div([inputs, html.Div()], className='row')

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.Div(
        html.H1(' - Prediction models - ', style={'color': colors['text'], 'fontSize': 36, 'text-align': 'center'})),
    html.Hr(
        style={'margin': '20px', 'padding': 0, 'height': '2px', 'border-top': '2px solid', 'border-bottom': '2px solid',
               'background': '#995E39'}),
    html.Div([
        # html.Label('Choose parameters:', style={'color': colors['text'], 'fontSize': 24, 'text-align': 'center'}),
        html.Div(inputs, className='col-md-2'),
        html.Div(className='col-md-1'),
        html.Div(style={'border-left': '3px solid black', 'height': '700px'}, className='col-md-2'),
        html.Div(dt.DataTable(id='result', columns=[{'id': 'MODEL', 'name': 'Model'},
                                                    {'id': 'PROB', 'name': 'Probability value'},
                                                    {'id': 'DIAG', 'name': 'Diagnosis'}]), className='col-md-4')],
        className='row')], className="container")
        # html.Div()

def roma(CA125, HE4):
    pi = -12 + 0.0626 * log(CA125) + 2.38 * log(HE4)
    roma = exp(pi) / (1 + exp(pi)) * 100
    return roma, roma > 7.4

def transform(AGE, CA125, HE4):
    Xn = mms.transform([[AGE, CA125, HE4]])
    Xten = Tensor(Xn)
    return Xten

def model(AGE, CA125, HE4):
    pred = mlpc(transform(AGE, CA125, HE4))
    Yv_pred = (pred[:, 1] > .5).numpy()
    return float(pred[:, 1]), Yv_pred

functions = {1: roma, 2: model}

@app.callback(Output('result', 'data'),
              [Input('button', 'n_clicks')],
              [State('AGE', 'value'), State('CA125', 'value'), State('HE4', 'value')])
def predict(n_clicks, AGE, CA125, HE4):     #n_clicks - количество нажатий на кнопку. если нажатий не было,значит ничего не возвращать, а возвращать только по нажатию
    if not n_clicks:
        return []
    for k, v in functions.items():
        if k == 1:
            c1, h1 = roma(CA125, HE4)
        elif k == 2:
            c2, h2 = model(AGE, CA125, HE4)
    if h1:
        h1 = 'Probability is high!'
    else:
        h1 = 'Low probability!'
    if h2:
        h2 = 'Probability is high!'
    else:
        h2 = 'Low probability!'
    return [{'MODEL': 'ROMA', 'PROB': "%.4f" % c1, 'DIAG': h1},
            {'MODEL': 'Neural Network', 'PROB': "%.4f" % c2, 'DIAG': h2}]
