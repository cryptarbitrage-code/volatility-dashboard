from api_functions import get_volatility_index_data
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd


# resolution of dvol data
now = datetime.now()
end_timestamp = round(datetime.timestamp(now) * 1000)
year_milliseconds = 1000 * 60 * 60 * 24 * 365
start_timestamp = end_timestamp - year_milliseconds
dvol_resolution = 3600 * 24  # resolution of vol data in seconds, e.g. 1 hour = 3600

def get_dvol_data():
    # fetch the DVOL data from the api
    btc_raw_data = get_volatility_index_data('BTC', start_timestamp, end_timestamp, dvol_resolution)
    eth_raw_data = get_volatility_index_data('ETH', start_timestamp, end_timestamp, dvol_resolution)

    # put btc data into a dataframe, add column names
    columns = ['timestamp', 'open', 'high', 'low', 'close']
    df = pd.DataFrame(btc_raw_data['data'], columns=columns)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['range'] = df['open'] - df['close']
    # eth dataframe
    df_eth = pd.DataFrame(eth_raw_data['data'], columns=columns)
    df_eth['date'] = pd.to_datetime(df_eth['timestamp'], unit='ms')
    df_eth['range'] = df_eth['open'] - df_eth['close']

    # vol stats
    current_vol = df.iloc[-1]['close']  # the last row (current candle) updates when fresh data is pulled
    current_vol_eth = df_eth.iloc[-1]['close']

    # IV Rank
    year_min = df['low'].min()
    year_max = df['high'].max()
    iv_rank = (current_vol - year_min) / (year_max - year_min) * 100
    # eth
    year_min_eth = df_eth['low'].min()
    year_max_eth = df_eth['high'].max()
    iv_rank_eth = (current_vol_eth - year_min_eth) / (year_max_eth - year_min_eth) * 100

    # IV percentile
    total_periods = len(df)
    periods_lower = len(df[(df['close'] <= current_vol)])
    iv_percentile = (periods_lower / total_periods) * 100
    # eth
    total_periods_eth = len(df_eth)
    periods_lower_eth = len(df_eth[(df_eth['close'] <= current_vol_eth)])
    iv_percentile_eth = (periods_lower_eth / total_periods_eth) * 100

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
        height=500, template='plotly_dark',
        title=f'BTC DVOL    High: {year_max}, Low: {year_min}, Current: {current_vol}'
    )

    # ETH DVOL chart
    candles_eth = go.Figure(
        data=[
            go.Candlestick(
                x=df_eth['date'],
                open=df_eth['open'],
                high=df_eth['high'],
                low=df_eth['low'],
                close=df_eth['close'],
            )
        ]
    )
    candles_eth.update_layout(
        height=500, template='plotly_dark',
        title=f'ETH DVOL    High: {year_max_eth}, Low: {year_min_eth}, Current: {current_vol_eth}'
    )

    # BTC/ETH DVOL ratio
    df_eth['ratio'] = df['close'] / df_eth['close']

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
    )


    return candles, iv_rank, iv_percentile, current_vol, year_min, year_max, \
           candles_eth, iv_rank_eth, iv_percentile_eth, current_vol_eth, year_min_eth, year_max_eth, ratio
