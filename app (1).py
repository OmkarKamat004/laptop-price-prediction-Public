# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Page config
st.set_page_config(page_title="Laptop Price Predictor 💻", page_icon="💻", layout="wide")
st.title("Laptop Price Predictor 💻")

# Load Data
df = pd.read_csv("laptop_data.csv")
st.dataframe(df)

# Clean Columns
df['Inches'] = df['Inches'].astype(float)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
df['X_res'] = df['ScreenResolution'].str.extract(r'(\d+)x').astype(float)
df['Y_res'] = df['ScreenResolution'].str.extract(r'x(\d+)').astype(float)
df['ppi'] = ((df['X_res'] ** 2 + df['Y_res'] ** 2) ** 0.5) / df['Inches']

def convert_memory(val):
    val = str(val).upper().replace(' ', '')
    hdd = 0
    ssd = 0
    if '+' in val:
        parts = val.split('+')
    else:
        parts = [val]

    for part in parts:
        if 'HDD' in part:
            if 'TB' in part:
                hdd += int(float(part.replace('HDD', '').replace('TB', '')) * 1000)
            elif 'GB' in part:
                hdd += int(float(part.replace('HDD', '').replace('GB', '')))
            else:
                hdd += int(float(part.replace('HDD', '')))
        elif 'SSD' in part:
            if 'TB' in part:
                ssd += int(float(part.replace('SSD', '').replace('TB', '')) * 1000)
            elif 'GB' in part:
                ssd += int(float(part.replace('SSD', '').replace('GB', '')))
            else:
                ssd += int(float(part.replace('SSD', '')))
    return pd.Series([hdd, ssd])

df[['HDD', 'SSD']] = df['Memory'].apply(convert_memory)

# Encode categorical features
company_unique = df['Company'].unique()
typename_unique = df['TypeName'].unique()
cpu_unique = df['Cpu'].unique()
gpu_unique = df['Gpu'].unique()
os_unique = df['OpSys'].unique()

le = LabelEncoder()
df['Company'] = le.fit_transform(df['Company'])
df['TypeName'] = le.fit_transform(df['TypeName'])
df['Cpu'] = le.fit_transform(df['Cpu'])
df['Gpu'] = le.fit_transform(df['Gpu'])
df['OpSys'] = le.fit_transform(df['OpSys'])

# Prepare input and output
X = df[['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS', 'ppi', 'Cpu', 'HDD', 'SSD', 'Gpu', 'OpSys']]
y = np.log(df['Price'])

# Train model
model = LinearRegression()
model.fit(X, y)

# Input form
st.header("Enter Laptop Features")

col1, col2, col3 = st.columns(3)
with col1:
    company_input = st.selectbox("Brand", company_unique)
    weight = st.number_input("Weight (kg)", step=0.1)
    touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])

with col2:
    typename_input = st.selectbox("Laptop Type", typename_unique)
    ips = st.selectbox("IPS Display", ['No', 'Yes'])
    ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024])

with col3:
    ram = st.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
    inches = st.number_input("Screen Size (inches)")
    hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048])

col4, col5 = st.columns(2)
with col4:
    cpu_input = st.selectbox("CPU Brand", cpu_unique)
with col5:
    gpu_input = st.selectbox("GPU Brand", gpu_unique)

os_input = st.selectbox("Operating System", os_unique)

if st.button("Predict Price"):
    company = np.where(company_unique == company_input)[0][0]
    typename = np.where(typename_unique == typename_input)[0][0]
    cpu = np.where(cpu_unique == cpu_input)[0][0]
    gpu = np.where(gpu_unique == gpu_input)[0][0]
    os = np.where(os_unique == os_input)[0][0]

    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    ppi = ((1920**2 + 1080**2) ** 0.5) / inches if inches > 0 else 0

    input_data = np.array([[company, typename, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]])
    prediction = model.predict(input_data)
    st.subheader(f"Estimated Price: Rs. {int(np.exp(prediction[0]))}")

# ------------------ Evaluation Section ------------------

st.header("📊 Model Evaluation & Visualizations")

# Residual Plot
y_pred = model.predict(X)
residuals = y - y_pred
fig, ax = plt.subplots()
sns.scatterplot(x=y_pred, y=residuals, ax=ax)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot")
st.pyplot(fig)

# Feature Correlation Heatmap
corr_matrix = df[['Price', 'Ram', 'Weight', 'Touchscreen', 'IPS', 'ppi', 'HDD', 'SSD']].corr()
fig2, ax2 = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax2)
ax2.set_title("Feature Correlation Heatmap")
st.pyplot(fig2)

# Actual vs Predicted Plot
actual_price = np.exp(y)
predicted_price = np.exp(y_pred)
fig3, ax3 = plt.subplots()
sns.scatterplot(x=actual_price, y=predicted_price, ax=ax3)
ax3.set_xlabel("Actual Price")
ax3.set_ylabel("Predicted Price")
ax3.set_title("Actual vs Predicted Price")
st.pyplot(fig3)

# Metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)

st.subheader("📈 Model Performance Metrics")
st.metric("R² Score", round(r2, 3))
st.metric("MAE", round(mae, 3))
st.metric("MSE", round(mse, 3))
