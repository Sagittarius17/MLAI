import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from cryptocmd import CmcScraper

# initialise scraper without time interval for max historical data
# scraper = CmcScraper("BTC")
# # Pandas dataFrame for the same data
# df = scraper.get_dataframe()

# # get raw data as list of list
# headers, data = scraper.get_data()

# # get data in a json format
# json_data = scraper.get_data("json")

# # export the data to csv
# scraper.export("csv", name="BTC_USD")

# # Load the data
# data = pd.read_csv('BTC_USD.csv')

# # Split the data into training and testing sets
# X = data[['Open', 'High', 'Low']]
# y = data['Close']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = model.predict(X_test)

# # Evaluate the performance of the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print('Mean squared error:', mse)
# print('R-squared:', r2)




import pandas as pd
import requests

# Define API parameters
url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': ''
}
params = {
    'start':'1',
    'limit':'1',
    'convert':'USD',
    'symbol':'BTC'
}

# Send request to CoinMarketCap API and parse the response
response = requests.get(url, headers=headers, params=params)
raw_data = response.json()

# Extract BTC data from the response
btc_data = raw_data['data'][0]
btc_price = btc_data['quote']['USD']['price']
btc_volume_24h = btc_data['quote']['USD']['volume_24h']
btc_market_cap = btc_data['quote']['USD']['market_cap']

# Create a Pandas dataframe with the BTC data
btc_df = pd.DataFrame({
    'Symbol': ['BTC'],
    'Price': [btc_price],
    'Volume_24h': [btc_volume_24h],
    'Market_Cap': [btc_market_cap]
})

# Save the dataframe to a CSV file
btc_df.to_csv('btc_data.csv', index=False)
