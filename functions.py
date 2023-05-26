from api_functions import get_volatility_index_data, get_book_summary_by_currency
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import dash_daq as daq
import pytz
import numpy as np
from scipy.stats import norm

# put this code inside the get_dvol_data function?
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


def calculate_time_difference(date_string):
    now = datetime.now(pytz.utc)
    date = datetime.strptime(date_string, "%d%b%y")
    date = date.replace(tzinfo=pytz.utc)
    target_time = date.replace(hour=8, minute=0, second=0)
    time_difference = (target_time - now).total_seconds()
    seconds_in_a_year = 365 * 24 * 60 * 60
    time_difference_years = time_difference / seconds_in_a_year
    return time_difference_years


def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

def calculate_implied_volatility(option_price, S, K, T, r):
    MAX_ITERATIONS = 100
    PRECISION = 0.0001
    sigma_low = 0.01
    sigma_high = 5
    implied_volatility = None

    for i in range(MAX_ITERATIONS):
        sigma = (sigma_low + sigma_high) / 2.0
        price = bs_call(S, K, T, r, sigma)
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
    btc_data = get_book_summary_by_currency(currency, 'option')
    df = pd.DataFrame(btc_data)
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
        0
    ), axis=1)
    # Group the DataFrame by 'expiry'
    grouped = df.groupby('expiry_date')
    # Apply the 'find_closest_strike' function to each group and collect the results
    df_term_structure = grouped.apply(find_closest_strike)
    # Reset the index and drop the original index column
    df_term_structure = df_term_structure.reset_index(drop=True)
    print(df_term_structure)

    # Create a line chart using Plotly
    fig = go.Figure(data=go.Scatter(
        x=df_term_structure['expiry_date'],
        y=df_term_structure['implied_volatility'],
        mode='lines'
    ))
    # Customize the layout
    fig.update_layout(
        title=f'{currency} Implied Volatility Term Structure',
        xaxis=dict(title='Expiry Date'),
        yaxis=dict(title='Implied Volatility'),
        template='plotly_dark',
    )

    return fig


def create_daq_gauge(gauge_id, color, minimum, maximum, label, value, size):
    gauge = daq.Gauge(
        id=gauge_id,
        color=color,
        showCurrentValue=True,
        min=minimum,
        max=maximum,
        label={'label': label, 'style': {'font-size': '20px'}},
        scale={'custom': {
            '0': {'label': '0', 'style': {'font-size': '15px'}},
            '20': {'label': '20', 'style': {'font-size': '15px'}},
            '40': {'label': '40', 'style': {'font-size': '15px'}},
            '60': {'label': '60', 'style': {'font-size': '15px'}},
            '80': {'label': '80', 'style': {'font-size': '15px'}},
            '100': {'label': '100', 'style': {'font-size': '15px'}},
        }},
        value=value,
        size=size,
    )

    return gauge
