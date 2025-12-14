# =========================================================
# STREAMLIT APP
# CLUSTERING PER TRANSAKSI + REGRESI ENSEMBLE
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Analisis Toko Bangunan", layout="wide")
st.title("📊 Clustering & Regresi Ensemble")
st.write("Analisis transaksi penjualan toko bangunan berbasis data")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv")

df = load_data()

st.subheader("📁 Dataset Awal")
st.dataframe(df.head())

# =========================================================
# CLUSTERING PER TRANSAKSI
# =========================================================
st.header("🔹 Clustering Transaksi (Per Transaksi)")

# 1. Ubah data per barang → per transaksi
df_cluster = df.groupby("ID Transaksi").agg({
    "Total Harga": "sum",
    "Kuantitas": "sum"
}).reset_index()

st.subheader("📊 Eksplorasi Data Transaksi")
st.write(df_cluster.describe())

# 2. Normalisasi
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_cluster[["Total Harga", "Kuantitas"]])

# 3. KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster["Cluster"] = kmeans.fit_predict(data_scaled)

st.subheader("📌 Jumlah Transaksi per Cluster")
st.write(df_cluster["Cluster"].value_counts())

# 4. Visualisasi clustering
fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.scatter(
    df_cluster["Total Harga"],
    df_cluster["Kuantitas"],
    c=df_cluster["Cluster"],
    alpha=0.7
)
ax1.set_xlabel("Total Belanja (Rupiah)")
ax1.set_ylabel("Jumlah Barang")
ax1.set_title("Segmentasi Transaksi Toko Bangunan")
st.pyplot(fig1)

# 5. Interpretasi cluster
st.subheader("📈 Profil Rata-rata Setiap Cluster")
cluster_mean = df_cluster.groupby("Cluster")[["Total Harga", "Kuantitas"]].mean()
st.write(cluster_mean.round(0))

# 6. Evaluasi clustering
score = silhouette_score(data_scaled, df_cluster["Cluster"])
st.write(f"**Silhouette Score:** {score:.4f}")
st.caption("Semakin mendekati 1, semakin baik pemisahan cluster")

# =========================================================
# REGRESI ENSEMBLE
# =========================================================
st.header("🔹 Regresi Ensemble (Prediksi Total Harga)")

df_reg = df.copy()

# Encode data kategorikal
cat_cols = ["Produk", "Kategori", "Satuan"]
encoder = LabelEncoder()
for col in cat_cols:
    df_reg[col] = encoder.fit_transform(df_reg[col])

# Drop kolom tidak relevan
df_reg = df_reg.drop(columns=["ID Transaksi", "Tanggal Pembelian"])

X = df_reg.drop(columns=["Total Harga"])
y = df_reg["Total Harga"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler_reg = StandardScaler()
X_train = scaler_reg.fit_transform(X_train)
X_test  = scaler_reg.transform(X_test)

# Model ensemble
rf  = RandomForestRegressor(n_estimators=200, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=200, random_state=42)

ensemble = VotingRegressor([
    ("RandomForest", rf),
    ("GradientBoosting", gbr)
])

# Training
ensemble.fit(X_train, y_train)

# Prediksi
y_pred = ensemble.predict(X_test)

# ---------------------------------------------------------
# Evaluasi Regresi
# ---------------------------------------------------------
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

st.subheader("📈 Hasil Evaluasi Regresi")
st.write(f"**MAE**  : {mae:.2f}")
st.write(f"**MSE**  : {mse:.2f}")
st.write(f"**RMSE** : {rmse:.2f}")
st.write(f"**R²**   : {r2:.4f}")

# ---------------------------------------------------------
# Visualisasi Regresi
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.scatter(y_test, y_pred, alpha=0.6)
ax2.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
ax2.set_xlabel("Nilai Aktual (Total Harga)")
ax2.set_ylabel("Nilai Prediksi")
ax2.set_title("Aktual vs Prediksi (Regresi Ensemble)")
st.pyplot(fig2)

st.success("✅ Analisis clustering dan regresi ensemble berhasil dijalankan.")
