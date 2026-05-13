# ==========================================
# SALES FORECASTING USING PROPHET
# Brazilian E-Commerce Dataset
# ==========================================

# STEP 1 — Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# STEP 2 — Load Dataset
df = pd.read_csv("Brazilian E-Commerce Public Dataset by Olist.csv")

# STEP 3 — Convert Date Column
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# STEP 4 — Extract Required Columns
sales_data = df[['order_purchase_timestamp', 'payment_value']]

# STEP 5 — Group Monthly Sales
monthly_sales = sales_data.groupby(
    pd.Grouper(
        key='order_purchase_timestamp',
        freq='D'
    )
)['payment_value'].sum().reset_index()

# STEP 6 — Rename Columns for Prophet
monthly_sales.columns = ['ds', 'y']

# Prophet requires:
# ds → date column
# y → target column

print(monthly_sales.head())

# STEP 7 — Create Prophet Model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True,changepoint_prior_scale=0.5)

# STEP 8 — Train Model
model.fit(monthly_sales)

# STEP 9 — Create Future Dates
future = model.make_future_dataframe(periods=6, freq='D')

# STEP 10 — Predict Future Sales
forecast = model.predict(future)

# STEP 11 — Display Forecast Results
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# STEP 12 — Plot Forecast
fig1 = model.plot(forecast)
plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# STEP 13 — Plot Trend Components
fig2 = model.plot_components(forecast)
plt.show()


from sklearn.metrics import mean_absolute_error

actual = monthly_sales['y']
predicted = forecast['yhat'][:len(monthly_sales)]

mae = mean_absolute_error(actual, predicted)

print("MAE:", mae)


# Create final forecast dataframe
final_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

# Merge actual sales
final_forecast = final_forecast.merge(
    monthly_sales,
    on='ds',
    how='left'
)

# Rename actual sales column
final_forecast.rename(columns={'y': 'Actual Sales'}, inplace=True)

# Export final CSV
final_forecast.to_csv("sales_forecast.csv", index=False)

print(final_forecast.head())






#https://learn.microsoft.com/en-us/training/modules/clean-data-power-bi/2-shape-data


#https://learn.microsoft.com/en-us/training/modules/clean-data-power-bi/6-profile-data