import json
import requests
from settings import api_exchange_address


def get_volatility_index_data(currency, start_timestamp, end_timestamp, resolution):
    url = "/api/v2/public/get_volatility_index_data"
    parameters = {
        'currency': currency,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp,
        'resolution': resolution,
    }
    # send HTTPS GET request
    # print(parameters)
    json_response = requests.get((api_exchange_address + url + "?"), params=parameters)
    response_dict = json.loads(json_response.content)
    vol_index_data = response_dict["result"]
    # print(vol_index_data)

    return vol_index_data

def get_book_summary_by_currency(currency, kind):
    url = "/api/v2/public/get_book_summary_by_currency"
    parameters = {
        'currency': currency,
        'kind': kind,
    }
    # send HTTPS GET request
    # print(parameters)
    json_response = requests.get((api_exchange_address + url + "?"), params=parameters)
    response_dict = json.loads(json_response.content)
    book_summary = response_dict["result"]
    # print(book_summary)

    return book_summary
