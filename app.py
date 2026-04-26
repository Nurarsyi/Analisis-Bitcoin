import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Konfigurasi Halaman (WAJIB PALING ATAS)
st.set_page_config(
    page_title="Bitcoin Analytics Dashboard",
    page_icon="₿",
    layout="wide"
)

# 2. Styling CSS (Biar tampilan Android/Dark Mode Premium)
st.markdown("""
    <style>
    .stApp { background: linear-gradient(to bottom, #121212, #1e1e1e); color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #1a1a1a; border-right: 1px solid #333; }
    div[data-testid="stMetric"] { 
        background-color: #262626; border: 1px solid #333; padding: 20px; 
        border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); 
    }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; background-color: #262626; border-radius: 10px 10px 0px 0px; color: #888; 
    }
    .stTabs [aria-selected="true"] { background-color: #f7931a !important; color: white !important; }
    .stButton>button { 
        background: linear-gradient(45deg, #f7931a, #ffab40); color: white; 
        border-radius: 12px; font-weight: bold; width: 100%; 
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Header
with st.container():
    col_logo, col_text = st.columns([1, 4])
    with col_logo:
        st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=80)
    with col_text:
        st.title("Bitcoin Intelligence")
        st.caption("Advanced Prediction Dashboard for Crypto Analysis")
st.divider()

# 4. Sidebar: Upload File Duluan
with st.sidebar:
    st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=100)
    st.title("₿ Control Center")
    uploaded_file = st.file_uploader("Upload Dataset Bitcoin Lo", type=["csv", "xlsx"])
    st.divider()

# 5. Logika Utama (Hanya jalan kalau file sudah diupload)
if uploaded_file is not None:
    try:
        # Baca Data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Pembersihan Data Otomatis
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                try:
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace('$', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except: pass
        
        df = df.dropna(how='all', axis=0)
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if date_cols:
            df = df.sort_values(by=date_cols[0])

        # --- ISI SIDEBAR SETELAH DF ADA ---
        with st.sidebar:
            st.subheader("⚙️ Data Settings")
            num_rows = st.slider("Tampilkan baris data:", 5, len(df), min(100, len(df)))
            
            st.subheader("🤖 AI Info")
            st.info("Model: Linear Regression")
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Clean Data", data=csv, file_name='bitcoin_data.csv')

        # --- TAMPILAN TAB ---
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Chart", "🤖 Prediction", "✨ Insight"])

        with tab1:
            st.subheader("Ringkasan Data")
            # Tampilkan metrics harga terakhir jika ada kolom angka
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                c1, c2 = st.columns(2)
                c1.metric("Data Points", len(df))
                c2.metric("Columns", len(df.columns))
            
            st.dataframe(df.head(num_rows), use_container_width=True)

        with tab2:
            st.subheader("Visualisasi Interaktif")
            all_cols = df.columns.tolist()
            c1, c2, c3 = st.columns(3)
            chart_type = c1.selectbox("Jenis Grafik", ["Line Chart", "Bar Chart", "Scatter Plot"])
            x_v = c2.selectbox("Sumbu X", all_cols)
            y_v = c3.selectbox("Sumbu Y", num_cols if num_cols else all_cols)

            if chart_type == "Line Chart": fig = px.line(df, x=x_v, y=y_v)
            elif chart_type == "Bar Chart": fig = px.bar(df, x=x_v, y=y_v)
            else: fig = px.scatter(df, x=x_v, y=y_v)
            
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("🤖 Bitcoin Price Prediction")
            if len(num_cols) >= 2:
                col_a, col_b = st.columns(2)
                feat_x = col_a.selectbox("Fitur (X):", num_cols, key="x_pred")
                target_y = col_b.selectbox("Target (Y):", num_cols, index=len(num_cols)-1, key="y_pred")

                X = df[[feat_x]].values
                y = df[target_y].values
                model = LinearRegression().fit(X, y)
                
                val = st.number_input(f"Input Nilai {feat_x}:", value=float(df[feat_x].iloc[-1]))
                if st.button("Prediksi Sekarang"):
                    res = model.predict([[val]])
                    st.metric("Hasil Prediksi", f"${res[0]:,.2f}")
                    st.balloons()
            else:
                st.warning("Butuh minimal 2 kolom angka!")

        with tab4:
            st.subheader("✨ Auto-Insight")
            if num_cols:
                st.success(f"💡 Kolom dengan nilai tertinggi saat ini adalah: **{df[num_cols].max().idxmax()}**")
                st.write("Gunakan data ini untuk melihat korelasi antara volume dan harga closing.")

    except Exception as e:
        st.error(f"Waduh ada error: {e}")

else:
    st.info("👋 Halo! Silakan upload file dataset Bitcoin (.csv atau .xlsx) di sidebar untuk memulai.")