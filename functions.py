from api_functions import get_volatility_index_data, get_book_summary_by_currency
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import pytz
import numpy as np
from scipy.stats import norm
import math
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html

N = norm.cdf
Np = norm.pdf

def hv_charts(currency_data, time_frames):

    # Create a figure for close-to-close volatility
    fig_close = px.line(currency_data, x=currency_data.index,
                        y=[f'{days}_day_close_vol' for days in time_frames],
                        height=400,
                        template='plotly_dark',
                        )
    fig_close.update_layout(hovermode='x unified')

    for trace in fig_close.data:
        trace.hovertemplate = '%{y:.4f}<extra></extra>'

    # Create a figure for Parkinson volatility
    fig_park = px.line(currency_data, x=currency_data.index,
                       y=[f'{days}_day_park_vol' for days in time_frames],
                       height=400,
                       template='plotly_dark',
                       )
    fig_park.update_layout(hovermode='x unified')

    # Create a figure for Close:Parkinson ratio
    fig_close_park_ratio = px.line(currency_data, x=currency_data.index,
                                   y=[f'{days}_day_park_close_ratio' for days in time_frames],
                                   height=400,
                                   template='plotly_dark',
                                   )
    fig_close_park_ratio.update_layout(hovermode='x unified')
    # Adjust the hovertemplate for each trace to only show y-values
    for trace in fig_close.data:
        trace.hovertemplate = '%{y:.4f}<extra></extra>'
    for trace in fig_park.data:
        trace.hovertemplate = '%{y:.4f}<extra></extra>'
    for trace in fig_close_park_ratio.data:
        trace.hovertemplate = '%{y:.4f}<extra></extra>'


    # Update traces to set visibility
    traces_to_show = [7, 30, 365]  # periods to show by default

    for i, period in enumerate(time_frames):
        if period not in traces_to_show:
            fig_close.data[i].visible = 'legendonly'

    for i, period in enumerate(time_frames):
        if period not in traces_to_show:
            fig_park.data[i].visible = 'legendonly'

    for i, period in enumerate(time_frames):
        if period not in traces_to_show:
            fig_close_park_ratio.data[i].visible = 'legendonly'

    # volatility cones
    percentiles = [10, 50, 90]
    percentile_colors = {10: 'MediumPurple', 50: 'MediumSeaGreen', 90: 'MediumPurple'}
    df_hv_btc_cleaned = currency_data.dropna()

    # Calculate percentiles for each window length
    volatility_percentiles = {
        window: {perc: np.percentile(df_hv_btc_cleaned[f'{window}_day_park_vol'], perc) for perc in percentiles} for window in
        time_frames}

    fig_vol_cones = go.Figure()

    # Plot each percentile
    for perc in percentiles:
        fig_vol_cones.add_trace(go.Scatter(
            x=time_frames,
            y=[volatility_percentiles[window][perc] for window in time_frames],
            mode='lines+markers',
            name=f'{perc}th percentile',
            line=dict(color=percentile_colors[perc])
        ))

    # Update the layout as needed
    fig_vol_cones.update_layout(
        xaxis_title="Window Length (days)",
        yaxis_title="Volatility",
        template="plotly_dark",
        hovermode='x unified'
    )

    # Calculate min and max for each window length
    volatility_min_max = {window: {'min': np.min(df_hv_btc_cleaned[f'{window}_day_park_vol']),
                                   'max': np.max(df_hv_btc_cleaned[f'{window}_day_park_vol'])}
                          for window in time_frames}

    # Add min and max to the plot
    fig_vol_cones.add_trace(go.Scatter(
        x=time_frames,
        y=[volatility_min_max[window]['min'] for window in time_frames],
        mode='lines+markers',
        name='Min',
        line=dict(color='crimson')
    ))

    fig_vol_cones.add_trace(go.Scatter(
        x=time_frames,
        y=[volatility_min_max[window]['max'] for window in time_frames],
        mode='lines+markers',
        name='Max',
        line=dict(color='crimson')
    ))

    return fig_close, fig_park, fig_close_park_ratio, fig_vol_cones


def dvol_charts(currency, start_timestamp, end_timestamp, dvol_resolution):
    #get the dvol data from the deribit api
    raw_data = get_volatility_index_data(currency, start_timestamp, end_timestamp, dvol_resolution)

    # put btc data into a dataframe, add column names
    columns = ['timestamp', 'open', 'high', 'low', 'close']
    df = pd.DataFrame(raw_data['data'], columns=columns)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['range'] = df['open'] - df['close']

    # vol stats
    current_vol = df.iloc[-1]['close']  # the last row (current candle) updates when fresh data is pulled

    # IV Rank
    year_min = df['low'].min()
    year_max = df['high'].max()
    iv_rank = (current_vol - year_min) / (year_max - year_min) * 100

    # IV percentile
    total_periods = len(df)
    periods_lower = len(df[(df['close'] <= current_vol)])
    iv_percentile = (periods_lower / total_periods) * 100

    # BTC DVOL chart
    candles = go.Figure(
        data=[
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
            )
        ]
    )
    candles.update_layout(
        height=400,
        template='plotly_dark',
        title=f'{currency} DVOL    High: {year_max}, Low: {year_min}, Current: {current_vol}',
        shapes=[
            dict(
                type='line',
                yref='y', y0=current_vol, y1=current_vol,
                xref='x', x0=df['date'].min(), x1=df['date'].max(),
                line=dict(
                    color='magenta',
                    width=1,
                    dash='dot',
                )
            )
        ]
    )

    return df, current_vol, iv_rank, iv_percentile, candles

def get_dvol_data():
    # resolution of dvol data
    now = datetime.now()
    end_timestamp = round(datetime.timestamp(now) * 1000)
    year_milliseconds = 1000 * 60 * 60 * 24 * 365
    start_timestamp = end_timestamp - year_milliseconds
    dvol_resolution = 3600 * 24  # resolution of vol data in seconds, e.g. 1 hour = 3600

    # calculate the dvol statistics and charts
    df_btc, btc_current_vol, btc_iv_rank, btc_iv_percentile, btc_candles = dvol_charts('BTC', start_timestamp, end_timestamp, dvol_resolution)
    df_eth, eth_current_vol, eth_v_rank, eth_iv_percentile, eth_candles = dvol_charts('ETH', start_timestamp, end_timestamp, dvol_resolution)

    # BTC/ETH DVOL ratio
    df_eth['ratio'] = df_btc['close'] / df_eth['close']

    ratio = go.Figure(
        go.Scatter(
            x=df_eth['date'],
            y=df_eth['ratio']
        )
    )
    ratio.update_layout(
        title="BTC/ETH DVOL Ratio",
        xaxis_title="Date",
        yaxis_title="Ratio",
        template='plotly_dark',
        height=400
    )

    return btc_candles, btc_iv_rank, btc_iv_percentile, eth_candles, eth_v_rank, eth_iv_percentile, ratio


def calculate_time_difference(date_string):
    now = datetime.now(pytz.utc)
    date = datetime.strptime(date_string, "%d%b%y")
    date = date.replace(tzinfo=pytz.utc)
    target_time = date.replace(hour=8, minute=0, second=0)
    time_difference = (target_time - now).total_seconds()
    seconds_in_a_year = 365 * 24 * 60 * 60
    time_difference_years = time_difference / seconds_in_a_year
    return time_difference_years


def bs_price(S, K, T, R, sigma, option_type):
    d1 = (np.log(S / K) + (R + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        price = S * N(d1) - K * np.exp(-R*T)* N(d2)
    elif option_type == "P":
        price = K*np.exp(-R*T)*N(-d2) - S*N(-d1)
    return price

def bs_delta(S, K, T, R, sigma, option_type):
    d1 = (np.log(S / K) + (R + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    if option_type == "C":
        delta = N(d1)
    elif option_type == "P":
        delta = N(d1) - 1
    return delta

def calculate_implied_volatility(option_price, S, K, T, R, option_type):
    MAX_ITERATIONS = 100
    PRECISION = 0.0001
    sigma_low = 0.01
    sigma_high = 5
    implied_volatility = None

    for i in range(MAX_ITERATIONS):
        sigma = (sigma_low + sigma_high) / 2.0
        price = bs_price(S, K, T, R, sigma, option_type)
        diff = option_price - price

        if abs(diff) < PRECISION:
            implied_volatility = sigma
            break

        if diff > 0:
            sigma_low = sigma
        else:
            sigma_high = sigma

    return implied_volatility

def find_closest_strike(row):
    # finds the strike closest to the underlying price (ATM)
    idx = np.abs(row['strike'] - row['underlying_price']).idxmin()
    return row.loc[idx, ['strike', 'expiry_date', 'implied_volatility']]

def vol_term_structure(currency):
    data = get_book_summary_by_currency(currency, 'option')
    df = pd.DataFrame(data)
    df = df[['underlying_price', 'mark_price', 'instrument_name']]
    df[['currency', 'expiry', 'strike', 'type']] = df['instrument_name'].str.split('-', expand=True)
    df = df.drop(df[df['type'] == 'P'].index)
    df['usd_price'] = df['underlying_price'] * df['mark_price']
    df['strike'] = df['strike'].astype(float)
    # Drop rows where underlying_price is more than 15% away from strike
    df = df.drop(df[abs(df['underlying_price'] - df['strike']) / df['strike'] > 0.15].index)
    df['time_to_expiry'] = df['expiry'].apply(lambda x: calculate_time_difference(x))
    df['expiry_date'] = pd.to_datetime(df['expiry'], format='%d%b%y')
    # Apply the calculate_implied_volatility function to create the 'implied_volatility' column
    df['implied_volatility'] = df.apply(lambda row: calculate_implied_volatility(
        row['usd_price'],
        row['underlying_price'],
        row['strike'],
        row['time_to_expiry'],
        0,
        row['type']
    ), axis=1)
    # Group the DataFrame by 'expiry'
    grouped = df.groupby('expiry_date')
    # Apply the 'find_closest_strike' function to each group and collect the results
    df_term_structure = grouped.apply(find_closest_strike)
    # Reset the index and drop the original index column
    df_term_structure = df_term_structure.reset_index(drop=True)

    # Create a line chart using Plotly
    fig = go.Figure(data=go.Scatter(
        x=df_term_structure['expiry_date'],
        y=df_term_structure['implied_volatility'],
        mode='lines',
        # line_shape='spline'  # enable this for a smoothed line
    ))
    # Generate vertical lines for each expiry_date
    shapes = []
    for expiry_date in df_term_structure['expiry_date']:
        shapes.append(
            dict(
                type='line',
                xref='x', x0=expiry_date, x1=expiry_date,
                yref='y', y0=df_term_structure['implied_volatility'].min(),
                y1=df_term_structure['implied_volatility'].max(),
                line=dict(
                    color='rgba(255, 0, 255, 0.5)',
                    width=1,
                    dash='dash',
                )
            )
        )
    # Customize the layout
    fig.update_layout(
        title=f'{currency} Implied Volatility Term Structure',
        xaxis=dict(title='Expiry Date'),
        yaxis=dict(title='Implied Volatility'),
        template='plotly_dark',
        shapes=shapes,
        height=400,
    )

    return fig

def vol_surface(currency):
    data = get_book_summary_by_currency(currency, 'option')
    df = pd.DataFrame(data)
    df = df[['underlying_price', 'mark_price', 'instrument_name']]
    df[['currency', 'expiry', 'strike', 'type']] = df['instrument_name'].str.split('-', expand=True)
    df = df.drop(df[df['type'] == 'P'].index)
    df['expiry_date'] = pd.to_datetime(df['expiry'], format='%d%b%y')
    df['usd_price'] = df['underlying_price'] * df['mark_price']
    df['strike'] = df['strike'].astype(float)
    df['time_to_expiry'] = df['expiry'].apply(lambda x: calculate_time_difference(x))
    df['expiry_date'] = pd.to_datetime(df['expiry'], format='%d%b%y')
    # Apply the calculate_implied_volatility function to create the 'implied_volatility' column
    df['implied_volatility'] = df.apply(lambda row: calculate_implied_volatility(
        row['usd_price'],
        row['underlying_price'],
        row['strike'],
        row['time_to_expiry'],
        0,
        row['type']
    ), axis=1)
    df['delta'] = df.apply(lambda row: bs_delta(
        row['underlying_price'],
        row['strike'],
        row['time_to_expiry'],
        0,
        row['implied_volatility'],
        row['type']
    ), axis=1)

    # drop extremes of delta
    df = df[df['delta'] >= 0.01]
    df = df[df['delta'] <= 0.99]

    fig = go.Figure(data=go.Scatter3d(
        x=df['expiry_date'],
        y=df['delta'],
        z=df['implied_volatility'],
        mode='markers',
        marker=dict(
            size=3,
            color=df['implied_volatility'],  # Color code based on 'implied_volatility' values
            colorscale='Sunset_r',  # Choose a colorscale
            opacity=0.8
        ),
        hovertemplate=
        '<b>Expiry Date:</b>: %{x}' +
        '<br><b>Delta:</b>: %{y}' +
        '<br><b>Implied Volatility:</b>: %{z}<br>' +
        '<extra></extra>', # removes the secondary box
    ))
    # round the maximum vol for the z axis to the nearest 0.2
    max_vol = df['implied_volatility'].max()
    rounded_max_vol = math.ceil(max_vol / 0.2) * 0.2
    fig.update_layout(
        title=f'{currency} 3D Volatility Surface',
        scene=dict(
            xaxis_title='Expiry Date',
            yaxis_title='Delta',
            zaxis_title='Implied Volatility',
            zaxis=dict(range=[0, rounded_max_vol], dtick=0.2)
        ),
        template='plotly_dark',
        height=800,
    )

    return fig

def draw_indicator(color, minimum, maximum, title, value, width, height):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [minimum, maximum], 'dtick': 20},
               'bar': {'color': color}}
    ))
    fig.update_layout(
        template='plotly_dark',
        autosize=False,
        margin=dict(
            l=30,  # left margin
            r=40,  # right margin
            b=0,  # bottom margin
            t=20,  # top margin
            pad=0  # padding
        ),
        paper_bgcolor="rgba(0,0,0,0)",  # makes the background transparent
        height=height,
        width=width,
    )
    return fig

def chart_card(title, chart, info_text):
    card = dbc.Card(
        children=[
            dbc.CardHeader(title, style={'padding-left': '50px'}),
            dbc.CardBody(chart, style={'padding': '0px'}),
            dbc.Badge(
                html.B("i"),
                color="primary",
                id=f'{chart.id}_info',
                pill=True,
                style={"position": "absolute", "top": "10px", "left": "20px", "zIndex": 2}
            ),
            dbc.Tooltip(
                info_text,
                target=f'{chart.id}_info',
            ),
        ],
        style={
            'width': '49%',
            'display': 'inline-block',
            'min-width': '600px',
            'margin': '2px',
        }
    )
    return card
