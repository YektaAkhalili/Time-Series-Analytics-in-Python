import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

df = pd.read_csv('data1.csv', parse_dates=["Date"], index_col=0)
# print(df.head(5))
df_agg = df.Total.resample('D').sum()
decomposition = seasonal_decompose(df_agg, model='additive')

# decomposition.plot()
# plt.show()

# ----------- test ------------
# null hypo : non-stationary

from statsmodels.tsa import stattools
from statsmodels.tsa import seasonal
adf_result = stattools.adfuller(df["Total"], autolag='AIC')
# print('p-val of the ADF test on irregular variations in CPI series data:', adf_result[1])

# p-val is  6.152823829962283e-16 : H0 is correct, it is non-stationary

# plots ACF and PACF

from pandas.plotting import autocorrelation_plot
# _ = autocorrelation_plot(df["Total"], color='b')
# plt.show()

#  ---------------ARIMA modeling----------------
df.index = pd.to_datetime(df.index)
# df.columns = ["Total"]
# print(df.head())

import pmdarima
from pmdarima.arima import auto_arima
stepwise_model = auto_arima(df, start_p=1, start_q=1,max_p=5, max_q=5, m=7,start_P=0, seasonal=True, d=0, D=1, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)
print(stepwise_model.aic())

# Best one is:  ARIMA: order=(0, 0, 0) seasonal_order=(1, 1, 1, 12);
# Best one is: order=(1, 0, 1) seasonal_order=(0, 1, 0, 7);


from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(df, test_size=0.3, random_state=42)
#
# # train_ = df.iloc[0:20]
# # test_ = df.iloc[21:]
#
stepwise_model.fit(x_train)
forecast1 = stepwise_model.predict(n_periods=15)
print(forecast1)

# forecast1 = pd.DataFrame(forecast1, index= x_test.index , columns=['Forecast'])
# pd.concat([x_test, forecast1], axis=1).iplot()
