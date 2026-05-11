import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("Brazilian E-Commerce Public Dataset by Olist.csv")

df['order_purchase_timestamp'] = pd.to_datetime(
    df['order_purchase_timestamp']
)

customer_data = df[
    ['order_purchase_timestamp', 'customer_unique_id']
]

daily_customers = customer_data.groupby(
    pd.Grouper(
        key='order_purchase_timestamp',
        freq='D'
    )
)['customer_unique_id'].nunique().reset_index()

daily_customers.columns = ['ds', 'y']

print(daily_customers.head())

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.5
)

model.fit(daily_customers)

future = model.make_future_dataframe(
    periods=30,
    freq='D'
)

forecast = model.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig1 = model.plot(forecast)

plt.title("Customer Demand Forecast")

plt.xlabel("Date")

plt.ylabel("Number of Customers")

plt.show()

fig2 = model.plot_components(forecast)

plt.show()


actual = daily_customers['y']

predicted = forecast['yhat'][:len(daily_customers)]

mae = mean_absolute_error(
    actual,
    predicted
)

print("MAE:", mae)

final_forecast = forecast[
    ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
].copy()

final_forecast = final_forecast.merge(
    daily_customers,
    on='ds',
    how='left'
)

final_forecast.rename(
    columns={'y': 'Actual Customers'},
    inplace=True
)

final_forecast.to_csv(
    "customer_demand_forecast.csv",
    index=False
)
print(final_forecast.head())

print("CSV Exported Successfully")