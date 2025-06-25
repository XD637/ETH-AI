import requests

url = 'https://api.coingecko.com/api/v3/coins/ethereum/market_chart'
params = {'vs_currency': 'usd', 'days': '3650'}  # 10 years
response = requests.get(url, params=params)
data = response.json()

print(data)  # Inspect the full response
