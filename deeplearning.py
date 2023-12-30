import pandas_datareader.data as web
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error

# Collects the data from the database
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#print("DEBUG USE 201: STEP 1");

#btc = web.get_data_yahoo(['BTC-USD'], start=datetime.datetime(2018, 1, 1), end=datetime.datetime(2020, 12, 2))['Close']
btc = web.get_data_yahoo(['BTC-USD'], start=datetime.datetime(2017, 12, 31), end=datetime.datetime(2020, 12, 2))

#print("DEBUG USE 202: STEP 2");
#print(btc.head())

# Formats the data by date
btc = pd.read_csv("BTC-USD2.csv")
btc.index = pd.to_datetime(btc['Date'], format='%Y-%m-%d')
del btc['Date']
#print(btc.head())

# Sets up the axis for the graph
sns.set()
plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.xlim(datetime.datetime(2018, 1, 1), datetime.datetime(2020, 12, 2))

#      Establishes the y-limits
with open("BTC-USD2.csv") as data_file:
	arrData = data_file.readlines()

del arrData[0]
for data in arrData:
	oneDataList = data.split(",")
	if oneDataList[0] == "2018-01-01":
		dataMinIdx = arrData.index(data)
	elif oneDataList[0] == "2020-12-02":
		dataMaxIdx = arrData.index(data)
	oneDataList.clear();
iMin = -1.0
iMax = -1.0
for x in range(dataMinIdx, dataMaxIdx):
	oneDataList = arrData[x].split(",")
	adj_close = float(oneDataList[6].rstrip("\n"))
	if adj_close < iMin:
		iMin = adj_close
	elif adj_close > iMax:
		iMax = adj_close
	oneDataList.clear();

plt.ylim(iMin, iMax)

# Splits data for training and testing
train = btc[btc.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
train = train[train.index >= pd.to_datetime("2018-01-01", format='%Y-%m-%d')]
test = btc[btc.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
test = test[test.index <= pd.to_datetime("2020-12-02", format='%Y-%m-%d')]
print(train['Adj Close'])

# -------ARMA MODEL-------
y = train['Adj Close'] #defines input
ARMAmodel = SARIMAX(y, order = (1, 0, 1)) #first value = autoregressive average specification [the number of lag observations in the model, also known as the lag order]; second value = integration/shift order [the number of times the raw observations are differenced; also known as the degree of differencing]; third value = moving average specification [the size of the moving average window, also known as the order of the moving average]
ARMAmodel = ARMAmodel.fit() #fits the model

# Generates predictions
arma_y_pred = ARMAmodel.get_forecast(len(test.index))
arma_y_pred_df = arma_y_pred.conf_int(alpha = 0.05) 
arma_y_pred_df["Predictions"] = ARMAmodel.predict(start = arma_y_pred_df.index[0], end = arma_y_pred_df.index[-1])
arma_y_pred_df.index = test.index
arma_y_pred_out = arma_y_pred_df["Predictions"] 

# -------ARIMA MODEL-------
ARIMAmodel = ARIMA(y, order = (5, 4, 2)) #uses input for ARMA Model; originally (2, 2, 2) but data was on top of ARMA data, so increased difference parameter to (5, 4, 2)
ARIMAmodel = ARIMAmodel.fit()

# Generates predictions
arima_y_pred = ARIMAmodel.get_forecast(len(test.index))
arima_y_pred_df = arima_y_pred.conf_int(alpha = 0.05) 
arima_y_pred_df["Predictions"] = ARIMAmodel.predict(start = arima_y_pred_df.index[0], end = arima_y_pred_df.index[-1])
#print(arima_y_pred_df.index[-1])
arima_y_pred_df.index = test.index
arima_y_pred_out = arima_y_pred_df["Predictions"]
#print(arima_y_pred_out)

# -------SARIMAX MODEL-------
#print("STEP 1");
SARIMAXmodel = SARIMAX(y, order = (5, 4, 2), seasonal_order=(2,2,2,12))
SARIMAXmodel = SARIMAXmodel.fit()

# Generates predictions
#print("STEP 2");
sarimax_y_pred = SARIMAXmodel.get_forecast(len(test.index))
sarimax_y_pred_df = sarimax_y_pred.conf_int(alpha = 0.05)
#print("sarimax_y_pred_df.index[0]", sarimax_y_pred_df.index[0]); 
#print("sarimax_y_pred_df.index[-1]", sarimax_y_pred_df.index[-1]); 
sarimax_y_pred_df["Predictions"] = SARIMAXmodel.predict(start = sarimax_y_pred_df.index[0], end = sarimax_y_pred_df.index[-1])
sarimax_y_pred_df.index = test.index
sarimax_y_pred_out = sarimax_y_pred_df["Predictions"]


print("STEP 3");
# Displays data, colorcoating the training and testing data
plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.plot(train, color = "black", label = 'Training')
plt.plot(test, color = "red", label = 'Testing')
plt.plot(arma_y_pred_out, color='green', label = 'ARMA Predictions')
plt.plot(arima_y_pred_out, color='yellow', label = 'ARIMA Predictions')
plt.plot(sarimax_y_pred_out, color='Blue', label = 'SARIMA Predictions')
#         addresses the issue with duplicate legend labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

#plt.plot(btc.index, btc['Adj Close'])
#plt.xticks(rotation=45)
#plt.title("Train/Test split for BTC Data")

# Evaluate the models using root mean-squared error
arma_rmse = np.sqrt(mean_squared_error(test["Adj Close"].values, arma_y_pred_df["Predictions"]))
print("ARMA RMSE: ",arma_rmse)
arima_rmse = np.sqrt(mean_squared_error(test["Adj Close"].values, arima_y_pred_df["Predictions"]))
print("ARIMA RMSE: ",arima_rmse)
sarimax_rmse = np.sqrt(mean_squared_error(test["Adj Close"].values, sarimax_y_pred_df["Predictions"]))
print("SARIMA RMSE: ",sarimax_rmse)
plt.show() #plots the data
