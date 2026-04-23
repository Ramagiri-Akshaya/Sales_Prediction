import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")

st.title("📊 Retail Sales Forecasting")

# Sidebar
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.sidebar.success("File uploaded successfully!")

    # -------- COLUMN SELECTION --------
    date_col = st.sidebar.selectbox("Select Date Column", df.columns)
    sales_col = st.sidebar.selectbox("Select Sales Column", df.columns)
    category_col = st.sidebar.selectbox("Select Category Column", df.columns)

    # -------- CATEGORY OPTIONS --------
    category_options = ["ALL"] + list(df[category_col].dropna().unique())
    selected_category = st.sidebar.selectbox("Select Category", category_options)

    future_days = st.sidebar.selectbox("Future Days to Predict", [7, 15, 30], index=2)

    if st.sidebar.button("Train Model"):

        # -------- FILTER CATEGORY --------
        if selected_category != "ALL":
            df = df[df[category_col] == selected_category]

        # -------- PREPROCESS --------
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)

        df = df.groupby(date_col)[sales_col].sum().reset_index()

        # -------- TITLE --------
        if selected_category == "ALL":
            st.subheader("📊 Forecast for All Categories")
        else:
            st.subheader(f"📊 Forecast for Category: {selected_category}")

        data = df[[sales_col]].values

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # -------- CREATE DATASET --------
        def create_dataset(data, time_step=30):
            X, y = [], []
            for i in range(len(data) - time_step):
                X.append(data[i:i + time_step])
                y.append(data[i + time_step])
            return np.array(X), np.array(y)

        time_step = 30
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # -------- MODEL --------
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        st.success("✅ Model Trained Successfully!")

        # -------- PREDICTIONS --------
        train_predict = model.predict(X)
        train_predict = scaler.inverse_transform(train_predict)
        actual = scaler.inverse_transform(y)

        actual = actual.flatten()
        train_predict = train_predict.flatten()

        # -------- METRICS --------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sales", f"{int(df[sales_col].sum()):,}")
        col2.metric("Average Daily Sales", f"{df[sales_col].mean():.2f}")
        col3.metric("Max Daily Sales", f"{df[sales_col].max()}")
        col4.metric("Min Daily Sales", f"{df[sales_col].min()}")

        # -------- ACTUAL VS PREDICTED --------
        st.subheader("📈 Actual vs Predicted Sales")

        dates = df[date_col].iloc[time_step:]

        fig1, ax1 = plt.subplots(figsize=(10,5))
        ax1.plot(dates, actual, label="Actual")
        ax1.plot(dates, train_predict, label="Predicted")

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Sales")
        ax1.legend()

        st.pyplot(fig1)

        # -------- FUTURE FORECAST --------
        st.subheader(f"🔮 Next {future_days} Days Forecast")

        last_data = scaled_data[-time_step:]
        future_input = last_data.reshape(1, time_step, 1)

        future_preds = []

        for _ in range(future_days):
            pred = model.predict(future_input, verbose=0)[0][0]
            future_preds.append(pred)

            new_input = np.append(future_input[0][1:], [[pred]], axis=0)
            future_input = new_input.reshape(1, time_step, 1)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

        future_dates = pd.date_range(df[date_col].iloc[-1], periods=future_days+1)[1:]

        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.plot(df[date_col], df[sales_col], label="Historical")
        ax2.plot(future_dates, future_preds, label="Future", color='green')

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Sales")
        ax2.legend()

        st.pyplot(fig2)

        # -------- INSIGHTS --------
        st.subheader("💡 Insights & Decision")

        trend = "increasing" if df[sales_col].iloc[-1] > df[sales_col].iloc[0] else "decreasing"

        df['Month'] = pd.to_datetime(df[date_col]).dt.to_period('M')
        monthly_sales = df.groupby('Month')[sales_col].sum()

        top_months = monthly_sales.sort_values(ascending=False).head(3).index

        sorted_months = sorted(top_months)
        start_month = sorted_months[0].strftime('%b')
        end_month = sorted_months[-1].strftime('%b')
        year = sorted_months[0].year

        peak_range = f"{start_month} to {end_month} {year}"

        # -------- INVENTORY DECISION --------
        last_sales = df[sales_col].iloc[-1]
        future_avg = future_preds.mean()

        if future_avg < last_sales:
            st.error("📉 Reduce Stock Storage")
        elif future_avg > last_sales:
            st.success("📈 Increase Stock")
        else:
            st.warning("➖ Maintain Stock")

        # -------- DISPLAY --------
        col1, col2, col3 = st.columns(3)
        col1.info(f"📈 Trend: {trend}")
        col2.info(f"📅 Peak Season: {peak_range}")
        col3.info(f"🔮 Forecast stable")

else:
    st.info("Upload dataset to start 🚀")