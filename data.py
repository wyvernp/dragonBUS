import datetime
from influx import InfluxDBHelper
from agile import OctopusEnergyPrices
from model import *
import pandas as pd
import pytz

def get_30min_usage_data(influx_host, influx_port, influx_username, influx_password, influx_database, first_agile_date, meter_id, timezone_str):
    # Read data from InfluxDB
    query = (
        f'SELECT sum("value") AS "sum_value" FROM "homeassistant"."autogen"."kWh" '
        f'WHERE time >= \'{first_agile_date}\' AND "entity_id"=\'{meter_id}\' '
        'GROUP BY time(30m) FILL(null)'
    )
    influx_helper = InfluxDBHelper(host=influx_host, port=influx_port, username=influx_username, password=influx_password, database=influx_database)
    result = influx_helper.query_data(query)

    raw_data = result['series'][0]['values']
    data = []

    # Set the timezone
    timezone = pytz.timezone(timezone_str)

    # Convert timestamps to datetime objects and process the mean consumption
    for point in raw_data:
        time = datetime.datetime.strptime(point[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).astimezone(timezone)
        sum_value = float(point[1]) if point[1] is not None else 0  # Handle None values
        data.append({'time': time, 'sum_value': sum_value})

    # Print the aggregated data
    usage_data = []
    for entry in data:
        formatted_time = entry['time'].astimezone(pytz.timezone(timezone_str)).strftime('%Y-%m-%d %H:%M:%S%z')
        usage_data.append([formatted_time, entry['sum_value']])
        #print(f"Time: {entry['time']}, Sum Value: {entry['sum_value']}")
    influx_helper.close_connection()
    return usage_data

def get_usage_model_data(config):
    usage = get_30min_usage_data(config['influx_host'], config['influx_port'], config['influx_username'], config['influx_password'], config['influx_database'], config['first_agile_date'], config['meter_id'], config['timezone'])
    first_date = usage[0][0]  # Accessing the first element's time
    last_date = usage[-1][0]  # Accessing the last element's time
    agile = OctopusEnergyPrices(config['product_code'], config['tariff_code'], config['agile_api_key'], config['timezone'])
    prices = agile.fetch_agile_prices(first_date, last_date)
    usage_df = pd.DataFrame(usage, columns=['usage_time', 'usage'])

    merged_data = prepare_data(usage_df, prices, config['timezone'])
    model = train_model(merged_data)
    return model

def predict_next_24_hours(config, model):
    # Ensure prices array is the correct length (48 half-hour slots)
    agile = OctopusEnergyPrices(config['product_code'], config['tariff_code'], config['agile_api_key'], config['timezone'])
    prices_df = agile.fetch_next_prices()
    
    # Extract the price values from the DataFrame
    prices = prices_df['value_inc_vat'].values
    
    # Current time in the specified timezone
    current_time = datetime.datetime.now(pytz.timezone(config['timezone']))
    # Adjust for the next full half-hour
    start_time = current_time.replace(minute=0, second=0, microsecond=0)
    
    if len(prices) < 48:
        prices = np.pad(prices, (0, 48 - len(prices)), 'edge')  # Pad with the last value
    elif len(prices) > 48:
        prices = prices[:48]  # Trim to the first 48 values

    future_times = pd.date_range(start=start_time, periods=48, freq='30T', tz=config['timezone'])
    future_data = pd.DataFrame({'time': future_times, 'price': prices})
    future_data['hour'] = future_data['time'].dt.hour
    future_data['day_of_week'] = future_data['time'].dt.dayofweek
    future_data['is_cheapest'] = 0

    # Identify the cheapest timeslot in the next 24 hours
    cheapest_timeslot = future_data.sort_values(by='price').iloc[0]['time']
    future_data.loc[future_data['time'] == cheapest_timeslot, 'is_cheapest'] = 1

    # Predict consumption
    future_X = future_data[['price', 'hour', 'day_of_week', 'is_cheapest']]
    future_data['predicted_consumption'] = model.predict(future_X)

    # Increase the consumption at the cheapest timeslot to simulate the water heater
    future_data.loc[future_data['time'] == cheapest_timeslot, 'predicted_consumption'] += 1  # Adjust this spike value as necessary

    return future_data[['time', 'predicted_consumption']]