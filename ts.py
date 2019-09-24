import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', parse_dates=["Date"], index_col=0)
print(df.head(5))
df_agg = df.Total.resample('M').sum()
decomposition = seasonal_decompose(df_agg, model='additive')

decomposition.plot()
plt.show()

# ----------- test ------------
# null hypothesis of ADF test : non-stationary

from statsmodels.tsa import stattools
from statsmodels.tsa import seasonal
adf_result = stattools.adfuller(df["Sales"], autolag='AIC')
print("p-value: ", adf_result[1])

# ---------- ACF and PACF ----------

from pandas.plotting import autocorrelation_plot
_ = autocorrelation_plot(df["Sales"], color='m')
plt.show()

# ---------------ARIMA modeling----------------
df.index = pd.to_datetime(df.index)

import pmdarima
from pmdarima.arima import auto_arima
stepwise_model = auto_arima(df, start_p=1, start_q=1,max_p=5, max_q=5, m=7,start_P=0, seasonal=True, d=0, D=1, trace=True,error_action='ignore',suppress_warnings=True,stepwise=True)
print(stepwise_model.aic())

# This gives a bunch of "recommendations" 

from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(df, test_size=0.3, random_state=42)

stepwise_model.fit(x_train)
forecast1 = stepwise_model.predict(n_periods=15)
print(forecast1)

