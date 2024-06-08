import requests
from datetime import datetime, timedelta
import pytz
import pandas as pd

class OctopusEnergyPrices:
    def __init__(self, product_code, tariff_code, api_key, timezone):
        self.product_code = product_code
        self.tariff_code = tariff_code
        self.api_key = api_key
        self.timezone = timezone

    def get_expensive_slots(segment, slotnum):
        sorted_segment = segment.sort_values(by='value_inc_vat', ascending=False)
        # Get the top 'slotnum' expensive timestamps
        top_timestamps = sorted_segment.head(slotnum)['valid_from'].tolist()
        return top_timestamps

    def get_cheap_slots(segment, slotnum):
        sorted_segment = segment.sort_values(by='value_inc_vat', ascending=True)
        # Get the top 'slotnum' cheap timestamps
        cheap_timestamps = sorted_segment.head(slotnum)['valid_from'].tolist()
        return cheap_timestamps

    def get_negative_slots(segment):
        # Filter the segment to only include rows with negative prices
        negative_prices_segment = segment[segment['value_inc_vat'] < 0]
        sorted_segment = negative_prices_segment.sort_values(by='value_inc_vat', ascending=False)
        negative_timestamps = sorted_segment['valid_from'].tolist()
        return negative_timestamps

    def fetch_next_prices(self):
        # Set time period to fetch prices from now to 24 hours from now
        current_time = datetime.now(pytz.timezone(self.timezone))
        period_from = current_time.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        period_to = (current_time + timedelta(hours=24)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

        # URL for the Agile tariff API endpoint
        url = f'https://api.octopus.energy/v1/products/{self.product_code}/electricity-tariffs/{self.tariff_code}/standard-unit-rates/'

        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        params = {
            'period_from': period_from,
            'period_to': period_to
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            prices = response.json()['results']
            if prices:
                simplified_prices = []
                for price in prices:
                    valid_from = datetime.strptime(price['valid_from'], '%Y-%m-%dT%H:%M:%SZ')
                    # Convert to the specified timezone
                    valid_from = valid_from.replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(self.timezone))
                    simplified_price = {
                        'valid_from': valid_from,
                        'value_inc_vat': price['value_inc_vat']
                    }
                    simplified_prices.append(simplified_price)
                prices = pd.DataFrame(simplified_prices)
                return prices
            else:
                print("No prices found for the specified period.")
        else:
            print(f"Failed to fetch prices: {response.status_code}")


    def fetch_agile_prices(self, start_date, end_date, chunk_size=1):
        headers = {
            'Authorization': self.api_key
        }
        
        date_range = pd.date_range(start_date, end_date, freq=f'{chunk_size}D')
        all_prices = pd.DataFrame()

        for i in range(len(date_range) - 1):
            period_from = date_range[i].strftime('%Y-%m-%dT%H:%M:%SZ')
            period_to = date_range[i + 1].strftime('%Y-%m-%dT%H:%M:%SZ')
            
            url = f'https://api.octopus.energy/v1/products/{self.product_code}/electricity-tariffs/{self.tariff_code}/standard-unit-rates/'
            params = {
                'period_from': period_from,
                'period_to': period_to
            }

            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                prices = pd.DataFrame(data['results'])
                prices['timestamp'] = pd.to_datetime(prices['valid_from'])
                prices.set_index('timestamp', inplace=True)
                prices.sort_index(inplace=True)
                prices['price'] = prices['value_inc_vat']
                all_prices = pd.concat([all_prices, prices[['price']]])
            else:
                raise Exception(f"Failed to fetch agile prices for {period_from} to {period_to}")

        # Reset the index to have the timestamp as the first column and price as the second column
        all_prices.reset_index(inplace=True)
        
        return all_prices
'''
agile = OctopusEnergyPrices('AGILE-18-02-21', 'E-1R-AGILE-18-02-21-J', 'api_key', 'Europe/London')
prices = agile.fetch_next_prices()
high_prices = OctopusEnergyPrices.get_expensive_slots(prices, slotnum=10)
cheap_prices = OctopusEnergyPrices.get_cheap_slots(prices, slotnum=10)
negative_prices = OctopusEnergyPrices.get_negative_slots(prices)
print(high_prices)
print(cheap_prices)
print(negative_prices)

'''