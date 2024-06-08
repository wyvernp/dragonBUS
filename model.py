import pandas as pd
import numpy as np
import pytz
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def prepare_data(usage_df, prices_df, local_timezone='Europe/London'):
    # Ensure the timestamps are in datetime format
    usage_df['usage_time'] = pd.to_datetime(usage_df['usage_time'])
    prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])

    # Define the timezone
    uk_timezone = pytz.timezone(local_timezone)

    # Handle timezone-naive datetime columns by localizing to the local timezone
    if usage_df['usage_time'].dt.tz is None:
        usage_df['usage_time'] = usage_df['usage_time'].dt.tz_localize(uk_timezone, ambiguous='NaT', nonexistent='NaT')
    else:
        usage_df['usage_time'] = usage_df['usage_time'].dt.tz_convert(uk_timezone)

    if prices_df['timestamp'].dt.tz is None:
        prices_df['timestamp'] = prices_df['timestamp'].dt.tz_localize(uk_timezone, ambiguous='NaT', nonexistent='NaT')
    else:
        prices_df['timestamp'] = prices_df['timestamp'].dt.tz_convert(uk_timezone)

    # Convert the datetime columns to UTC
    usage_df['usage_time'] = usage_df['usage_time'].dt.tz_convert('UTC')
    prices_df['timestamp'] = prices_df['timestamp'].dt.tz_convert('UTC')

    # Merge the dataframes on the timestamp column
    merged_data = pd.merge(usage_df, prices_df, left_on='usage_time', right_on='timestamp')

    # Drop the redundant 'timestamp' column after merging
    merged_data.drop(columns=['timestamp'], inplace=True)

    # Feature engineering
    merged_data['hour'] = merged_data['usage_time'].dt.hour
    merged_data['day_of_week'] = merged_data['usage_time'].dt.dayofweek
    merged_data['is_cheapest'] = 0

    # Determine the cheapest timeslot for each day and mark it
    for date, segment in merged_data.groupby(merged_data['usage_time'].dt.date):
        cheapest_timeslot = segment.sort_values(by='price').iloc[0]['usage_time']
        merged_data.loc[merged_data['usage_time'] == cheapest_timeslot, 'is_cheapest'] = 1

    return merged_data

def train_model(data):
    # Define features and target
    X = data[['price', 'hour', 'day_of_week', 'is_cheapest']]
    y = data['usage']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Gradient Boosting Regressor model
    model = GradientBoostingRegressor(n_estimators=40, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    
    return model

