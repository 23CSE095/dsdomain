import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation Dashboard")
st.markdown("""
Upload your cleaned customer CSV and explore **RFM analysis** and **K-Means clustering** interactively.  
Adjust Recency delay and number of clusters, and download the clustered data.
""")

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload cleanedcustomer.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Ensure correct types
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    # -----------------------------
    # Sidebar controls
    # -----------------------------
    st.sidebar.header("Configuration")
    delay_days = st.sidebar.slider("Recency Delay (days)", 1, 30, 1)
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 4)

    # -----------------------------
    # Compute RFM
    # -----------------------------
    NOW = df['InvoiceDate'].max() + pd.Timedelta(days=delay_days)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (NOW - x.max()).days,
        'InvoiceNo': 'count',
        'TotalAmount': 'sum'
    }).rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalAmount':'Monetary'}).reset_index()
    rfm = rfm[rfm['Monetary'] > 0].reset_index(drop=True)

    # -----------------------------
    # K-Means clustering
    # -----------------------------
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # -----------------------------
    # Show RFM Table
    # -----------------------------
    st.subheader("RFM Table with Clusters")
    st.dataframe(rfm)

    # -----------------------------
    # Scatter plot: Recency vs Monetary
    # -----------------------------
    st.subheader("Recency vs Monetary Scatter Plot")
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        x='Recency',
        y='Monetary',
        hue='Cluster',
        data=rfm,
        palette='Set1',
        s=100
    )
    plt.xlabel('Recency (days)')
    plt.ylabel('Monetary Value')
    plt.title('Customer Segments')
    plt.legend(title='Cluster')
    st.pyplot(plt)

    # -----------------------------
    # CSV download
    # -----------------------------
    csv = rfm.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download RFM CSV",
        data=csv,
        file_name='rfm_clusters.csv',
        mime='text/csv'
    )

else:
    st.info("Please upload your cleanedcustomer.csv file to begin analysis.")