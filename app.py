import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

st.title("Customer Segmentation Dashboard")
st.markdown("Perform **RFM Analysis** and **K-Means Clustering** easily.")

def preprocess_data(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    return df


def compute_rfm(df, delay_days):
    NOW = df['InvoiceDate'].max() + pd.Timedelta(days=delay_days)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (NOW - x.max()).days,
        'InvoiceNo': 'count',
        'TotalAmount': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    }).reset_index()

    return rfm[rfm['Monetary'] > 0]


def apply_kmeans(rfm, n_clusters):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    model = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = model.fit_predict(scaled)

    return rfm


def plot_clusters(rfm):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=rfm,
        x='Recency',
        y='Monetary',
        hue='Cluster',
        palette='Set2',
        s=80,
        ax=ax
    )
    ax.set_title("Customer Segments")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary Value")
    st.pyplot(fig)

st.sidebar.header("Settings")

delay_days = st.sidebar.slider("Recency Delay (days)", 1, 30, 1)
n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 4)
uploaded_file = st.file_uploader("Upload cleanedcustomer.csv", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df)

        rfm = compute_rfm(df, delay_days)
        rfm = apply_kmeans(rfm, n_clusters)
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Customers", len(rfm))
        col2.metric("Average Recency", int(rfm['Recency'].mean()))
        col3.metric("Total Revenue", int(rfm['Monetary'].sum()))
        st.subheader("RFM Table")
        st.dataframe(rfm, use_container_width=True)
        st.subheader("Cluster Visualization")
        plot_clusters(rfm)
        csv = rfm.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results",
            data=csv,
            file_name="rfm_clusters.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload your dataset to get started 🚀")
