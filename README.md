# 📊 Retail Sales Forecasting using LSTM

## 🚀 Overview
This project implements a Retail Sales Forecasting System using a Long Short-Term Memory (LSTM) deep learning model. The system analyzes historical sales data and predicts future sales trends. It also provides category-wise analysis, visual insights, and inventory recommendations through an interactive dashboard built using Streamlit.

---

## 🎯 Features
- Upload your own CSV dataset  
- Visualize sales trends  
- LSTM-based sales prediction  
- Future forecasting (7, 15, 30 days)  
- Category-wise forecasting (ALL / specific category)  
- Actual vs Predicted comparison  
- Insights (trend, peak season)  
- Inventory recommendations (Reduce / Increase / Maintain stock)  

---

## 🧠 Technologies Used
- Python  
- TensorFlow / Keras (LSTM)  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Streamlit  

---

## 📂 Project Structure
├── app.py  
└── README.md  

---

## 📊 Dataset Format
Date,Sales,Category  
2023-01-01,500,Electronics  
2023-01-02,600,Clothing  

- Date → Time column  
- Sales → Numeric values  
- Category → Optional (for category-wise analysis)  

---


### Run Application
streamlit run app.py  

---

## 📈 How It Works
1. Upload dataset  
2. Select date, sales, and category columns  
3. Choose category (ALL or specific)  
4. Train LSTM model  
5. View predictions, graphs, and insights  

---

## 🧪 Model Details
- Model: LSTM (Recurrent Neural Network)  
- Time Step: 30 days  
- Loss Function: Mean Squared Error  
- Optimizer: Adam  

---

## 📦 Output
- Sales trend visualization  
- Predicted vs Actual comparison  
- Future sales forecast  
- Business insights  
- Inventory decision  

---

## 🎯 Use Cases
- Retail demand forecasting  
- Inventory management  
- Business decision support  
- Trend analysis  

---

## 🚀 Future Enhancements
- Multi-store forecasting  
- Advanced models (GRU, Transformer)  
- Interactive charts (Plotly)  
- Cloud deployment  

---
 



---

⭐ If you like this project, give it a star on GitHub!
