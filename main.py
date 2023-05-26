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
from functions import get_dvol_data, vol_term_structure, create_daq_gauge

pd.set_option('display.max_columns', None)

# Initialize the app
app = dash.Dash(title="Volatility Dashboard", external_stylesheets=[dbc.themes.DARKLY])

btc_dvol_candles, btc_iv_rank, btc_iv_percentile, current_vol, year_min, year_max, \
    eth_dvol_candles, eth_iv_rank, eth_iv_percentile, eth_current_vol, eth_year_min, eth_year_max, dvol_ratio = get_dvol_data()


# DVOL tab layout
dvol_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Button(
                html.B("Refresh"),
                color="info",
                id="refresh_button",
                className="my-3",
                style={'width': '100px'}
            ),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Badge(
                html.B("i"),
                color="primary",
                id="btc_dvol_info",
                pill=True,
                style={"position": "absolute", "top": "10px", "left": "20px", "zIndex": 2}
            ),
            dbc.Tooltip(
                "Daily candle chart for the Deribit bitcoin volatility index for the previous year.",
                target="btc_dvol_info",
            ),
            dbc.Row([dcc.Graph(id='btc_dvol_candles', figure=btc_dvol_candles)]),
        ], width=10, style={'position': 'relative'}),
        dbc.Col([
            dbc.Row([
                create_daq_gauge('btc_iv_rank_gauge', "#00cfbe", 0, 100, 'IV Rank', btc_iv_rank, 150)
            ]),
            dbc.Row([
                create_daq_gauge('btc_iv_percentile_gauge', "#00cfbe", 0, 100, 'IV Percentile', btc_iv_percentile, 150)
            ])
        ], width=1)

    ]),
    dbc.Row([
        dbc.Col([
            dbc.Badge(
                html.B("i"),
                color="primary",
                id="eth_dvol_info",
                pill=True,
                style={"position": "absolute", "top": "10px", "left": "20px", "zIndex": 2}
            ),
            dbc.Tooltip(
                "Daily candle chart for the Deribit ethereum volatility index for the previous year.",
                target="eth_dvol_info",
            ),
            dbc.Row([dcc.Graph(id='eth_dvol_candles', figure=eth_dvol_candles)]),
        ], width=10, style={'position': 'relative'}),
        dbc.Col([
            dbc.Row([
                create_daq_gauge('eth_iv_rank_gauge', "#00cfbe", 0, 100, 'IV Rank', eth_iv_rank, 150)
            ]),
            dbc.Row([
                create_daq_gauge('eth_iv_percentile_gauge', "#00cfbe", 0, 100, 'IV Percentile', eth_iv_percentile, 150)
            ])
        ], width=1)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Badge(
                html.B("i"),
                color="primary",
                id="dvol_ratio_info",
                pill=True,
                style={"position": "absolute", "top": "10px", "left": "20px", "zIndex": 2}
            ),
            dbc.Tooltip(
                [html.P("The ratio between BTC DVOL and ETH DVOL. Calculated using close data from the daily candles."),
                 html.P("When ratio > 1, BTC vol is higher. When ratio < 1, ETH vol is higher.")],
                target="dvol_ratio_info",
            ),
            dbc.Row([dcc.Graph(id='dvol_ratio', figure=dvol_ratio)]),
        ], width=10, style={'position': 'relative'})
    ], className="mb-3")

], fluid=True)

term_structure_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Badge(
                html.B("i"),
                color="primary",
                id="btc_term_structure_info",
                pill=True,
                style={"position": "absolute", "top": "10px", "left": "20px", "zIndex": 2}
            ),
            dbc.Tooltip(
                html.P("At the money implied volatility per expiry."),
                target="btc_term_structure_info",
            ),
            dcc.Graph(id='btc_term_structure', figure=vol_term_structure('BTC'))
        ], width=6, style={'position': 'relative'}),
    ], className="my-3"),
    dbc.Row([
        dbc.Col([
            dbc.Badge(
                html.B("i"),
                color="primary",
                id="eth_term_structure_info",
                pill=True,
                style={"position": "absolute", "top": "10px", "left": "20px", "zIndex": 2}
            ),
            dbc.Tooltip(
                html.P("At the money implied volatility per expiry."),
                target="eth_term_structure_info",
            ),
            dcc.Graph(id='eth_term_structure', figure=vol_term_structure('ETH'))
        ], width=6, style={'position': 'relative'}),
    ], className="mb-3"),
], fluid=True)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([html.H2(children='Crypto Volatility Dashboard')])
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(children='''
            Historical and implied volatility statistics for bitcoin and ethereum.
            '''),
        ])
    ]),
    dbc.Tabs([
        dbc.Tab(
            dvol_tab,
            label='DVOL',
            tab_id='dvol_tab',
            activeTabClassName='fw-bold',
            active_label_style={"color": "#00CFBE"},
        ),
        dbc.Tab(
            term_structure_tab,
            label='Term Structure',
            tab_id='term_structure_tab',
            activeTabClassName='fw-bold',
            active_label_style={"color": "#00CFBE"},
        ),
    ], id="tabs_main", active_tab="dvol_tab"
    ),
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