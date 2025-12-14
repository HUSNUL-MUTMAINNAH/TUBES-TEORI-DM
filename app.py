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
st.title("📊 Clustering & Regresi Penjualan Toko Bangunan")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv")

st.subheader("🔍 Data Awal")
st.dataframe(df.head())

# ======================
# CLUSTERING
# ======================
st.header("🟢 Clustering Pelanggan")

df_cluster = df.groupby("ID Transaksi").agg({
    "Total Harga": "sum",
    "Kuantitas": "sum"
}).reset_index()

scaler_cluster = StandardScaler()
cluster_scaled = scaler_cluster.fit_transform(
    df_cluster[["Total Harga", "Kuantitas"]]
)

kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster["Cluster"] = kmeans.fit_predict(cluster_scaled)

col1, col2 = st.columns(2)
with col1:
    st.write("Jumlah Transaksi per Cluster")
    st.write(df_cluster["Cluster"].value_counts())

with col2:
    sil = silhouette_score(cluster_scaled, df_cluster["Cluster"])
    st.success(f"Silhouette Score: {sil:.4f}")

fig1, ax1 = plt.subplots()
sns.scatterplot(
    data=df_cluster,
    x="Total Harga",
    y="Kuantitas",
    hue="Cluster",
    palette="viridis",
    ax=ax1
)
ax1.set_title("Visualisasi Cluster")
st.pyplot(fig1)

st.subheader("📌 Profil Cluster")
st.dataframe(
    df_cluster.groupby("Cluster")[["Total Harga", "Kuantitas"]].mean().round(0)
)

# ======================
# REGRESI ENSEMBLE
# ======================
st.header("🔵 Regresi Ensemble")

df_reg = df.copy()
encoder = LabelEncoder()

for col in ["Produk", "Kategori", "Satuan"]:
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

rf = RandomForestRegressor(n_estimators=200, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)

model = VotingRegressor([
    ("RF", rf),
    ("GB", gbr)
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:,.0f}")
col2.metric("RMSE", f"{rmse:,.0f}")
col3.metric("R²", f"{r2:.4f}")

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.6)
ax2.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
ax2.set_title("Aktual vs Prediksi")
st.pyplot(fig2)

# ======================
# PREDIKSI MANUAL
# ======================
st.header("🧮 Prediksi Total Harga (Input Manual)")

col1, col2 = st.columns(2)

with col1:
    harga_satuan = st.number_input("Harga Satuan (Rp)", value=50000, step=1000)
    kuantitas = st.number_input("Jumlah Barang", value=4, min_value=1)

with col2:
    produk = st.selectbox("Produk", df["Produk"].unique())
    kategori = st.selectbox("Kategori", df["Kategori"].unique())
    satuan = st.selectbox("Satuan", df["Satuan"].unique())

if st.button("🔮 Hitung Prediksi"):
    produk_enc = LabelEncoder().fit(df["Produk"]).transform([produk])[0]
    kategori_enc = LabelEncoder().fit(df["Kategori"]).transform([kategori])[0]
    satuan_enc = LabelEncoder().fit(df["Satuan"]).transform([satuan])[0]

    input_data = np.array([[
        produk_enc,
        kategori_enc,
        satuan_enc,
        kuantitas,
        harga_satuan
    ]])

    input_scaled = scaler_reg.transform(input_data)
    hasil = model.predict(input_scaled)[0]

    st.success(f"💰 Prediksi Total Harga: **Rp {hasil:,.0f}**")
