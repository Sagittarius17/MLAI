import math
import nsepy as nse
import pandas as pd
import pandas_datareader as web
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

company = "ITC"
start_date = date(2020, 12, 15)
end_date = date.today()
print(end_date)

stock_data = nse.get_history(company, start=start_date, end=end_date)
# print(stock_data.head)

plt.plot(stock_data['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('ITC Closing Prices Tomorrow')
plt.show()