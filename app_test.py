import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def get_expensive_slots(segment, slotnum):
    sorted_segment = segment.sort_values(by='price (pence)', ascending=False)
    # Get the top 'slotnum' expensive timestamps
    top_timestamps = sorted_segment.head(slotnum)['Start'].tolist()
    return top_timestamps

def get_cheap_slots(segment, slotnum):
    sorted_segment = segment.sort_values(by='price (pence)', ascending=True)
    # Get the top 'slotnum' cheap timestamps
    cheap_timestamps = sorted_segment.head(slotnum)['Start'].tolist()
    return cheap_timestamps

def get_negative_slots(segment, slotnum):
    # Filter the segment to only include rows with negative prices
    negative_prices_segment = segment[segment['price (pence)'] < 0]
    sorted_segment = negative_prices_segment.sort_values(by='price (pence)', ascending=True)
    cheap_timestamps = sorted_segment.head(slotnum)['Start'].tolist()
    return cheap_timestamps


def prepare_data(consumption_path, agile_path):
    # Load the datasets
    consumption_data = pd.read_csv(consumption_path)
    agile_data = pd.read_csv(agile_path)

    # Ensure the timestamps are in datetime format
    consumption_data['Start'] = pd.to_datetime(consumption_data['Start'])
    agile_data['timestamp'] = pd.to_datetime(agile_data['timestamp'])

    # Merge the dataframes on the timestamp column
    merged_data = pd.merge(consumption_data, agile_data, left_on='Start', right_on='timestamp')

    # Drop the redundant 'timestamp' column after merging
    merged_data.drop(columns=['timestamp'], inplace=True)

    # Feature engineering
    merged_data['hour'] = merged_data['Start'].dt.hour
    merged_data['day_of_week'] = merged_data['Start'].dt.dayofweek
    merged_data['is_cheapest'] = 0

    # Determine the cheapest timeslot for each day and mark it
    for date, segment in merged_data.groupby(merged_data['Start'].dt.date):
        cheapest_timeslot = segment.sort_values(by='price (pence)').iloc[0]['Start']
        merged_data.loc[merged_data['Start'] == cheapest_timeslot, 'is_cheapest'] = 1

    return merged_data

def train_model(data):
    # Define features and target
    X = data[['price (pence)', 'hour', 'day_of_week', 'is_cheapest']]
    y = data['Consumption (kWh)']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Gradient Boosting Regressor model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    
    return model

def predict_next_24_hours(model, prices, start_time):
    # Ensure prices array is the correct length (48 half-hour slots)
    if len(prices) < 48:
        prices = np.pad(prices, (0, 48 - len(prices)), 'edge')  # Pad with the last value
    elif len(prices) > 48:
        prices = prices[:48]  # Trim to the first 48 values
    
    future_times = pd.date_range(start=start_time, periods=48, freq='30T')
    future_data = pd.DataFrame({'Start': future_times, 'price (pence)': prices})
    future_data['hour'] = future_data['Start'].dt.hour
    future_data['day_of_week'] = future_data['Start'].dt.dayofweek
    future_data['is_cheapest'] = 0
    
    # Identify the cheapest timeslot in the next 24 hours
    cheapest_timeslot = future_data.sort_values(by='price (pence)').iloc[0]['Start']
    future_data.loc[future_data['Start'] == cheapest_timeslot, 'is_cheapest'] = 1

    # Predict consumption
    future_X = future_data[['price (pence)', 'hour', 'day_of_week', 'is_cheapest']]
    future_data['predicted_consumption'] = model.predict(future_X)
    
    # Increase the consumption at the cheapest timeslot to simulate the water heater
    future_data.loc[future_data['Start'] == cheapest_timeslot, 'predicted_consumption'] += 1  # Adjust this spike value as necessary
    
    return future_data[['Start', 'predicted_consumption']]

# Add the path to your CSV files here
consumption_path = 'consumption1205-066.csv'
agile_path = 'csv_agile_J_South_Eastern_England m-j-test.csv'
slotnum = 6
# Prepare the data
merged_data = prepare_data(consumption_path, agile_path)
# Train the model
model = train_model(merged_data)
# Initialize a dataframe to hold all predictions
all_predictions = pd.DataFrame()

# Group by the new date segment starting at 17:00 each day
merged_data['DateSegment'] = (merged_data['Start'] - pd.Timedelta(hours=17)).dt.date
grouped_data = merged_data.groupby('DateSegment')

# Loop through each segment to predict the next 24 hours
for date, segment in grouped_data:
    print(f"Processing segment for date starting at 17:00 on {date}")
    start_time = segment['Start'].min()
    future_prices = segment['price (pence)'].values  # Use the prices from the segment for prediction
    predictions = predict_next_24_hours(model, future_prices, start_time)
    all_predictions = pd.concat([all_predictions, predictions])
    negative_slots = get_negative_slots(segment, slotnum)  # Adjust 'slotnum' as needed
    print(f"Negative slots for {date}: {negative_slots}")
    cheap_slots = get_cheap_slots(segment, slotnum)  # Adjust 'slotnum' as needed
    print(f"Cheap slots for {date}: {cheap_slots}")
    cheap_slots = get_expensive_slots(segment, slotnum=10)  # Adjust 'slotnum' as needed
    print(f"Exspensive slots for {date}: {cheap_slots}")

# Merge predictions with actual consumption data
result = pd.merge(merged_data, all_predictions, on='Start', how='left')

# Plot the actual vs predicted consumption for the entire dataset
plt.figure(figsize=(14, 7))
plt.plot(result['Start'], result['Consumption (kWh)'], label='Actual Consumption', color='blue')
plt.plot(result['Start'], result['predicted_consumption'], label='Predicted Consumption', color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Consumption (kWh)')
plt.title('Actual vs Predicted Consumption')
plt.legend()
plt.grid(True)
plt.show()
