import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

st.set_page_config(page_title="Clustering & Regresi Toko Bangunan", layout="wide")

st.title("📊 Analisis Clustering & Regresi Toko Bangunan")
st.markdown("Segmentasi pelanggan & prediksi total harga menggunakan **Ensemble Learning**")

# =========================
# LOAD DATA
# =========================
st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv")

st.subheader("🔍 Data Awal")
st.dataframe(df.head())
st.write("Jumlah data:", df.shape)

# =========================
# CLUSTERING
# =========================
st.header("🟢 Clustering Pelanggan (K-Means)")

df_cluster = df.groupby('ID Transaksi').agg({
    'Total Harga': 'sum',
    'Kuantitas': 'sum'
}).reset_index()

st.subheader("📈 Statistik Transaksi")
st.write(df_cluster.describe())

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_cluster[['Total Harga', 'Kuantitas']])

kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(data_scaled)

st.subheader("👥 Jumlah Transaksi per Cluster")
st.write(df_cluster['Cluster'].value_counts())

# Visualisasi cluster
fig1, ax1 = plt.subplots(figsize=(8,5))
sns.scatterplot(
    data=df_cluster,
    x='Total Harga',
    y='Kuantitas',
    hue='Cluster',
    palette='viridis',
    s=100,
    ax=ax1
)
ax1.set_title("Segmentasi Pelanggan")
ax1.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig1)

# Interpretasi
st.subheader("📌 Profil Cluster")
cluster_mean = df_cluster.groupby('Cluster')[['Total Harga', 'Kuantitas']].mean()
st.dataframe(cluster_mean.round(0))

sil_score = silhouette_score(data_scaled, df_cluster['Cluster'])
st.success(f"Silhouette Score: {sil_score:.4f}")

# =========================
# REGRESI
# =========================
st.header("🔵 Regresi Ensemble (Prediksi Total Harga)")

df_reg = df.copy()

cat_cols = ["Produk", "Kategori", "Satuan"]
encoder = LabelEncoder()
for col in cat_cols:
    df_reg[col] = encoder.fit_transform(df_reg[col])

df_reg = df_reg.drop(columns=["ID Transaksi", "Tanggal Pembelian"])

X = df_reg.drop(columns=["Total Harga"])
y = df_reg["Total Harga"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train = scaler_reg.fit_transform(X_train)
X_test = scaler_reg.transform(X_test)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

ensemble_model = VotingRegressor(
    estimators=[
        ("RandomForest", rf),
        ("GradientBoosting", gbr)
    ]
)

ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)

# Evaluasi
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Hasil Evaluasi")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:,.0f}")
col2.metric("RMSE", f"{rmse:,.0f}")
col3.metric("R²", f"{r2:.4f}")

# Visualisasi regresi
fig2, ax2 = plt.subplots(figsize=(6,5))
ax2.scatter(y_test, y_pred, alpha=0.6)
ax2.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
ax2.set_xlabel("Nilai Aktual")
ax2.set_ylabel("Nilai Prediksi")
ax2.set_title("Aktual vs Prediksi")
st.pyplot(fig2)

st.success("✅ Analisis Clustering & Regresi berhasil dijalankan")
