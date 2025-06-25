import requests
import pandas as pd

def fetch_eth_data(days=62):
    url = f"https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days={days}&interval=daily"
    response = requests.get(url)
    data = response.json()

    # Extracting close prices and volumes
    prices = data['prices']
    volumes = data['total_volumes']

    # Create a DataFrame
    df = pd.DataFrame(prices, columns=['Date', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['Volume'] = [v[1] for v in volumes]

    return df

# Fetch the last 62 days of ETH data
eth_data = fetch_eth_data(62)
print(eth_data)

csv_file_path = 'eth_data.csv'

# Save the DataFrame to a CSV file
eth_data.to_csv(csv_file_path, index=False)

print(f"ETH data saved to {csv_file_path}")
