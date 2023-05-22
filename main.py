# Volatility Dashboard
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from api_functions import get_volatility_index_data
from functions import get_dvol_data


# Initialize the app
app = dash.Dash(title="Volatility Dashboard", external_stylesheets=[dbc.themes.DARKLY])

btc_dvol_candles, btc_iv_rank, btc_iv_percentile, current_vol, year_min, year_max, \
eth_dvol_candles, eth_iv_rank, eth_iv_percentile, eth_current_vol, eth_year_min, eth_year_max, dvol_ratio = get_dvol_data()

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([html.H1(children='Crypto Volatility Dashboard')])
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(children='''
            Historical and implied volatility statistics for bitcoin and ethereum.
            '''),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button(
                html.B("Refresh"),
                color="info",
                id="refresh_button",
                className="mb-3",
                style={'width': '100px'}
            ),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([dcc.Graph(id='btc_dvol_candles', figure=btc_dvol_candles)]),
        ], width=10),
        dbc.Col([
            dbc.Row([
                daq.Gauge(
                    id='btc_iv_rank_gauge',
                    color="#00cfbe",
                    showCurrentValue=True,
                    min=0,
                    max=100,
                    label={'label': 'IV Rank', 'style': {'font-size': '20px'}},
                    scale={'custom': {
                        '0': {'label': '0', 'style': {'font-size': '15px'}},
                        '20': {'label': '20', 'style': {'font-size': '15px'}},
                        '40': {'label': '40', 'style': {'font-size': '15px'}},
                        '60': {'label': '60', 'style': {'font-size': '15px'}},
                        '80': {'label': '80', 'style': {'font-size': '15px'}},
                        '100': {'label': '100', 'style': {'font-size': '15px'}},
                    }},
                    value=btc_iv_rank,
                    size=150,
                )
            ]),
            dbc.Row([
                daq.Gauge(
                    id='btc_iv_percentile_gauge',
                    color="#00cfbe",
                    showCurrentValue=True,
                    min=0,
                    max=100,
                    label={'label': 'IV Percentile', 'style': {'font-size': '20px'}},
                    scale={'custom': {
                        '0': {'label': '0', 'style': {'font-size': '15px'}},
                        '20': {'label': '20', 'style': {'font-size': '15px'}},
                        '40': {'label': '40', 'style': {'font-size': '15px'}},
                        '60': {'label': '60', 'style': {'font-size': '15px'}},
                        '80': {'label': '80', 'style': {'font-size': '15px'}},
                        '100': {'label': '100', 'style': {'font-size': '15px'}},
                    }},
                    value=btc_iv_percentile,
                    size=150,
                )
            ])
        ], width=1)

    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([dcc.Graph(id='eth_dvol_candles', figure=eth_dvol_candles)]),
        ], width=10),
        dbc.Col([
            dbc.Row([
                daq.Gauge(
                    id='eth_iv_rank_gauge',
                    color="#00cfbe",
                    showCurrentValue=True,
                    min=0,
                    max=100,
                    label={'label': 'IV Rank', 'style': {'font-size': '20px'}},
                    scale={'custom': {
                        '0': {'label': '0', 'style': {'font-size': '15px'}},
                        '20': {'label': '20', 'style': {'font-size': '15px'}},
                        '40': {'label': '40', 'style': {'font-size': '15px'}},
                        '60': {'label': '60', 'style': {'font-size': '15px'}},
                        '80': {'label': '80', 'style': {'font-size': '15px'}},
                        '100': {'label': '100', 'style': {'font-size': '15px'}},
                    }},
                    value=eth_iv_rank,
                    size=150,
                )
            ]),
            dbc.Row([
                daq.Gauge(
                    id='eth_iv_percentile_gauge',
                    color="#00cfbe",
                    showCurrentValue=True,
                    min=0,
                    max=100,
                    label={'label': 'IV Percentile', 'style': {'font-size': '20px'}},
                    scale={'custom': {
                        '0': {'label': '0', 'style': {'font-size': '15px'}},
                        '20': {'label': '20', 'style': {'font-size': '15px'}},
                        '40': {'label': '40', 'style': {'font-size': '15px'}},
                        '60': {'label': '60', 'style': {'font-size': '15px'}},
                        '80': {'label': '80', 'style': {'font-size': '15px'}},
                        '100': {'label': '100', 'style': {'font-size': '15px'}},
                    }},
                    value=eth_iv_percentile,
                    size=150,
                )
            ])
        ], width=1)

    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([dcc.Graph(id='dvol_ratio', figure=dvol_ratio)]),
        ], width=10)
    ], className="mb-3")

], fluid=True)

@app.callback(
    [
        Output('btc_dvol_candles', 'figure'),
        Output('btc_iv_rank_gauge', 'value'),
        Output('btc_iv_percentile_gauge', 'value'),
        Output('eth_dvol_candles', 'figure'),
        Output('eth_iv_rank_gauge', 'value'),
        Output('eth_iv_percentile_gauge', 'value'),
        Output('dvol_ratio', 'figure'),
    ],
    Input('refresh_button', 'n_clicks'),
)
def refresh_data(n_clicks):
    print('button presses: ', n_clicks)

    btc_dvol_candles, btc_iv_rank, btc_iv_percentile, current_vol, year_min, year_max, eth_dvol_candles, \
    eth_iv_rank, eth_iv_percentile, eth_current_vol, eth_year_min, eth_year_max, dvol_ratio = get_dvol_data()

    return btc_dvol_candles, btc_iv_rank, btc_iv_percentile, eth_dvol_candles, eth_iv_rank, eth_iv_percentile, dvol_ratio

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)