# Volatility Dashboard
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from functions import get_dvol_data, vol_term_structure, vol_surface, draw_indicator

pd.set_option('display.max_columns', None)  # useful for testing

# Initialize the app
app = dash.Dash(title="Volatility Dashboard", external_stylesheets=[dbc.themes.DARKLY])

def fetch_data():
    # dvol data
    btc_dvol_candles, btc_iv_rank, btc_iv_percentile, eth_dvol_candles, eth_iv_rank, eth_iv_percentile, \
        dvol_ratio = get_dvol_data()
    # term structure
    btc_vol_term_structure = vol_term_structure('BTC')
    eth_vol_term_structure = vol_term_structure('ETH')
    # vol_surface
    btc_vol_surface = vol_surface('BTC')
    eth_vol_surface = vol_surface('ETH')
    return btc_dvol_candles, btc_iv_rank, btc_iv_percentile, eth_dvol_candles, eth_iv_rank, eth_iv_percentile, \
           dvol_ratio, btc_vol_term_structure, eth_vol_term_structure, btc_vol_surface, eth_vol_surface

btc_dvol_candles, btc_iv_rank, btc_iv_percentile, eth_dvol_candles, eth_iv_rank, eth_iv_percentile, \
    dvol_ratio, btc_vol_term_structure, eth_vol_term_structure, btc_vol_surface, eth_vol_surface = fetch_data()

# DVOL tab layout
dvol_tab = dbc.Container([
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
                dcc.Graph(id='btc_iv_rank_indicator', figure=draw_indicator('magenta', 0, 100, 'IV Rank', btc_iv_rank, 250, 200))
            ]),
            dbc.Row([
                dcc.Graph(id='btc_iv_percentile_indicator', figure=draw_indicator('magenta', 0, 100, 'IV Rank', btc_iv_percentile, 250, 200))
            ])
        ], width=2)

    ], className="my-3"),
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
                dcc.Graph(id='eth_iv_rank_indicator', figure=draw_indicator('magenta', 0, 100, 'IV Rank', eth_iv_rank, 250, 200))
            ]),
            dbc.Row([
                dcc.Graph(id='eth_iv_percentile_indicator', figure=draw_indicator('magenta', 0, 100, 'IV Rank', eth_iv_percentile, 250, 200))
            ])
        ], width=2)
    ], className="mb-3"),
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
                html.P("Current at the money implied volatility per expiry."),
                target="btc_term_structure_info",
            ),
            dcc.Graph(id='btc_vol_term_structure', figure=btc_vol_term_structure),
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
                html.P("Current at the money implied volatility per expiry."),
                target="eth_term_structure_info",
            ),
            dcc.Graph(id='eth_vol_term_structure', figure=eth_vol_term_structure)
        ], width=6, style={'position': 'relative'}),
    ], className="mb-3"),
], fluid=True)

vol_surface_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Badge(
                html.B("i"),
                color="primary",
                id="btc_vol_surface_info",
                pill=True,
                style={"position": "absolute", "top": "10px", "left": "20px", "zIndex": 2}
            ),
            dbc.Tooltip(
                [html.P("3D implied volatility surface for BTC."),
                 html.P("Deltas of <0.01 and >0.99 have been dropped.")],
                target="btc_vol_surface_info",
            ),
            dcc.Graph(id='btc_vol_surface', figure=btc_vol_surface)
        ], width=6, style={'position': 'relative'}),
        dbc.Col([
            dbc.Badge(
                html.B("i"),
                color="primary",
                id="eth_vol_surface_info",
                pill=True,
                style={"position": "absolute", "top": "10px", "left": "20px", "zIndex": 2}
            ),
            dbc.Tooltip(
                [html.P("3D implied volatility surface for ETH."),
                 html.P("Deltas of <0.01 and >0.99 have been dropped.")],
                target="eth_vol_surface_info",
            ),
            dcc.Graph(id='eth_vol_surface', figure=eth_vol_surface)
        ], width=6, style={'position': 'relative'}),
    ], className="my-3"),
], fluid=True)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([html.H3(children='Crypto Volatility Dashboard')]),
        dbc.Col([
            html.Div(
                dbc.Button(
                    html.B("Refresh"),
                    color="info",
                    id="refresh_button",
                    className="my-1",
                    style={'width': '100px'}
                ),
                style={'text-align': 'right'}
            ),
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
        dbc.Tab(
            vol_surface_tab,
            label='Vol Surface',
            tab_id='vol_surface_tab',
            activeTabClassName='fw-bold',
            active_label_style={"color": "#00CFBE"},
        ),
    ], id="tabs_main", active_tab="dvol_tab"
    ),
], fluid=True)

@app.callback(
    [
        Output('btc_dvol_candles', 'figure'),
        Output('btc_iv_rank_indicator', 'figure'),
        Output('btc_iv_percentile_indicator', 'figure'),
        Output('eth_dvol_candles', 'figure'),
        Output('eth_iv_rank_indicator', 'figure'),
        Output('eth_iv_percentile_indicator', 'figure'),
        Output('dvol_ratio', 'figure'),
        Output('btc_vol_term_structure', 'figure'),
        Output('eth_vol_term_structure', 'figure'),
        Output('btc_vol_surface', 'figure'),
        Output('eth_vol_surface', 'figure'),
    ],
    Input('refresh_button', 'n_clicks'),
)
def refresh_data(n_clicks):
    print('button presses: ', n_clicks)

    btc_dvol_candles, btc_iv_rank, btc_iv_percentile, eth_dvol_candles, eth_iv_rank, eth_iv_percentile, \
    dvol_ratio, btc_vol_term_structure, eth_vol_term_structure, btc_vol_surface, eth_vol_surface = fetch_data()

    btc_iv_rank_indicator = draw_indicator('magenta', 0, 100, 'IV Rank', btc_iv_rank, 250, 200)
    btc_iv_percentile_indicator = draw_indicator('magenta', 0, 100, 'IV Percentile', btc_iv_percentile, 250, 200)
    eth_iv_rank_indicator = draw_indicator('magenta', 0, 100, 'IV Rank', eth_iv_rank, 250, 200)
    eth_iv_percentile_indicator = draw_indicator('magenta', 0, 100, 'IV Percentile', eth_iv_percentile, 250, 200)

    return btc_dvol_candles, btc_iv_rank_indicator, btc_iv_percentile_indicator, eth_dvol_candles, \
           eth_iv_rank_indicator, eth_iv_percentile_indicator, dvol_ratio, btc_vol_term_structure, \
           eth_vol_term_structure, btc_vol_surface, eth_vol_surface

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)