import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessor import load_and_preprocess
from src.autoencoder_model import train_autoencoder
from src.explainer import compute_shap_insights

# 1. Page Configuration & Aesthetic Injection
st.set_page_config(page_title="ShieldFraud Enterprise Hub", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    /* Global Background and Typography Overrides */
    .stApp {
        background-color: #070A12;
        color: #F1F5F9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Navigation Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Custom High-Tech Card Containers */
    .metric-card {
        background: linear-gradient(145deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    
    /* Premium Interactive Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0EA5E9 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4) !important;
    }
    
    /* Tabs Selection Accent Override */
    button[data-baseweb="tab"] {
        font-size: 1rem !important;
        letter-spacing: 0.05rem !important;
        color: #94A3B8 !important;
    }
    button[aria-selected="true"] {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Master App Navigation Header
st.markdown("""
    <div style='display: flex; align-items: center; margin-bottom: 25px;'>
        <div style='background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%); padding: 12px; border-radius: 10px; margin-right: 15px;'>
            <span style='font-size: 24px;'>🛡️</span>
        </div>
        <div>
            <h1 style='margin: 0; font-size: 2.2rem; letter-spacing: -0.05rem;'>SHIELD-FRAUD SYSTEM <span style='color:#0EA5E9; font-size:1.2rem; vertical-align:middle;'>v2.0 PRO</span></h1>
            <p style='margin: 0; color: #64748B; font-size: 0.95rem;'>Autonomous Transaction Guard and Deep Anomaly Explainer Backend</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# 3. Persistent Caching & Session Architecture
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.device = None

@st.cache_data
def load_base_data_cache():
    import os
    data_path = 'data/creditcard.csv'
    
    if os.path.exists(data_path):
        # Local Mode: Load the real file if it exists on your machine
        X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
        X_train_normal = X_train[y_train == 0].sample(40000, random_state=42)
        return X_train, X_test, y_train, y_test, X_train_normal
    else:
        # Cloud Mode Fallback: Mathematically simulate dataset structure if file is missing from GitHub
        st.sidebar.info("☁️ Cloud Environment Active: Simulating baseline tensors...")
        
        np.random.seed(42)
        total_mock_records = 50000
        
        # Mirror the exact 29-dimensional feature layout (V1-V28 + scaled_amount)
        mock_features = np.random.normal(loc=0.0, scale=1.0, size=(total_mock_records, 29))
        feature_names = [f'V{i}' for i in range(1, 29)] + ['scaled_amount']
        
        X_train = pd.DataFrame(mock_features, columns=feature_names)
        X_test = pd.DataFrame(np.random.normal(loc=0.0, scale=1.0, size=(10000, 29)), columns=feature_names)
        
        y_train = pd.Series(np.zeros(total_mock_records))
        y_test = pd.Series(np.zeros(10000))
        
        return X_train, X_test, y_train, y_test, X_train

# 4. Control Panel Sidebar - Ingestion Source Toggle
st.sidebar.markdown("### 🕹️ DATA INGESTION SOURCE")
source_mode = st.sidebar.radio("Select Ingestion Stream Type:", ["📁 Static CSV Upload", "🗄️ Enterprise SQL Database"])

test_df = None

if source_mode == "📁 Static CSV Upload":
    uploaded_file = st.sidebar.file_uploader("Ingest Transaction Stream (CSV)", type=["csv"])
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
else:
    st.sidebar.markdown("#### 🔑 Database Connection Config")
    db_host = st.sidebar.text_input("Host Address", value="localhost")
    db_port = st.sidebar.text_input("Port", value="5432")
    db_name = st.sidebar.text_input("Database Name", value="bank_ledger")
    
    if st.sidebar.button("🔌 Query Live Transaction Batch"):
        with st.spinner("Connecting to secure database pipeline..."):
            try:
                from sqlalchemy import create_engine
                engine = create_engine(f'postgresql://postgres:password@{db_host}:{db_port}/{db_name}')
                query = "SELECT * FROM ledger_transactions WHERE checked = False LIMIT 5000"
                test_df = pd.read_sql(query, engine)
                st.sidebar.success(f"📥 Successfully pulled {len(test_df)} unverified records!")
                st.session_state.live_db_data = test_df
            except Exception as e:
                st.sidebar.error(f"❌ Connection Failed: Check your local configurations.")

if source_mode == "🗄️ Enterprise SQL Database" and 'live_db_data' in st.session_state:
    test_df = st.session_state.live_db_data

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ PARAMETER BOUNDS")
sensitivity = st.sidebar.slider("Anomaly Alarm Sensitivity (%)", min_value=90.0, max_value=99.9, value=98.0, step=0.1)

if st.sidebar.button("🏗️ INITIALIZE PLATFORM CORE", use_container_width=True):
    with st.spinner("Compiling neural network weights onto memory matrices..."):
        _, _, _, _, X_train_normal = load_base_data_cache()
        model, device = train_autoencoder(X_train_normal, epochs=6, batch_size=512)
        st.session_state.trained_model = model
        st.session_state.device = device
        st.sidebar.success(f"🚀 Active on {str(device).upper()} Backend!")

# 5. Core Operational Tabs
tab_dashboard, tab_forensics, tab_performance = st.tabs([
    "📟 MISSION CONTROL", "🔍 FORENSIC ANALYSIS SANDBOX", "📈 PERFORMANCE LOGISTICS"
])

# --- TAB 1: MISSION CONTROL ---
with tab_dashboard:
    if test_df is not None:
        display_df = test_df.drop('Class', axis=1, errors='ignore')
        
        if st.session_state.trained_model is None:
            st.info("💡 Subsystems standard. Click 'Initialize Platform Core' inside the sidebar node to sync memory vectors.")
        else:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            proc_df = display_df.copy()
            if 'Amount' in proc_df.columns:
                proc_df['scaled_amount'] = scaler.fit_transform(proc_df['Amount'].values.reshape(-1, 1))
            proc_df = proc_df.drop([c for c in ['Time', 'Amount'] if c in proc_df.columns], axis=1, errors='ignore')
            
            # Predict values using model
            model = st.session_state.trained_model
            device = st.session_state.device
            X_tensor = torch.tensor(proc_df.values, dtype=torch.float32).to(device)
            
            model.eval()
            with torch.no_grad():
                reconstructed = model(X_tensor)
                mse_losses = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
                
            threshold = np.percentile(mse_losses, sensitivity)
            flagged_idx = np.where(mse_losses > threshold)[0]
            
            # KPI Layout Row
            st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            with kpi1:
                st.markdown(f"<div class='metric-card'><small style='color:#94A3B8;'>BATCH VOLUME</small><h2 style='color:#F8FAFC; margin:5px 0;'>{len(test_df):,}</h2></div>", unsafe_allow_html=True)
            with kpi2:
                alert_color = "#EF4444" if len(flagged_idx) > 0 else "#10B981"
                st.markdown(f"<div class='metric-card'><small style='color:#94A3B8;'>RISK HIGHLIGHTS</small><h2 style='color:{alert_color}; margin:5px 0;'>{len(flagged_idx):,}</h2></div>", unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"<div class='metric-card'><small style='color:#94A3B8;'>DECISION BARRIER</small><h2 style='color:#0EA5E9; margin:5px 0;'>{threshold:.4f}</h2></div>", unsafe_allow_html=True)
            with kpi4:
                st.markdown(f"<div class='metric-card'><small style='color:#94A3B8;'>GPU ACCELERATION</small><h2 style='color:#A855F7; margin:5px 0;'>{str(device).upper()}</h2></div>", unsafe_allow_html=True)
            
            # Main Dashboard Split Visuals
            col_chart, col_list = st.columns([5, 3])
            with col_chart:
                st.markdown("### 📈 BATCH VARIANCE PROFILE")
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor('#070A12')
                ax.set_facecolor('#0F172A')
                
                sns.histplot(mse_losses, bins=50, kde=True, color='#0EA5E9', ax=ax, stat="density", alpha=0.7)
                ax.axvline(threshold, color='#EF4444', linestyle='--', linewidth=2, label=f'Alarm Threshold ({threshold:.2f})')
                ax.set_xlabel("Reconstruction Loss Curve (MSE)", color='#94A3B8')
                ax.legend(facecolor='#1E293B', edgecolor='#334155')
                st.pyplot(fig)
                
            with col_list:
                st.markdown("### 🚨 PRIORITY INCIDENT QUEUE")
                anomaly_summary = test_df.iloc[flagged_idx].copy()
                anomaly_summary['Risk Weight'] = mse_losses[flagged_idx]
                anomaly_summary = anomaly_summary.sort_values(by='Risk Weight', ascending=False)
                st.dataframe(anomaly_summary[['Risk Weight']].head(10), use_container_width=True, height=330)
                
            # Storing outputs systematically in session variables for forensics tabs
            st.session_state.proc_df = proc_df
            st.session_state.flagged_idx = flagged_idx
            st.session_state.mse_losses = mse_losses
            st.session_state.test_df = test_df
    else:
        st.markdown("""
            <div style='background-color: #0F172A; border: 1px dashed #334155; padding: 60px; border-radius: 14px; text-align: center; margin-top: 30px;'>
                <h3 style='color: #64748B;'>SYSTEM AWAITING INGESTION FEED</h3>
                <p style='color: #475569;'>Please drop an operational transaction format snapshot or pull from the database panel to spin up live tracking telemetry charts.</p>
            </div>
        """, unsafe_allow_html=True)

# --- TAB 2: FORENSIC ANALYSIS SANDBOX ---
with tab_forensics:
    st.markdown("### 🔬 MACHINE LEARNING AUDITING SUITE")
    if 'flagged_idx' not in st.session_state or len(st.session_state.flagged_idx) == 0:
        st.info("💡 Awaiting live anomalies from Mission Control to populate deep forensic mapping analysis trees.")
    else:
        f_col1, f_col2 = st.columns([1, 3])
        with f_col1:
            selected_target = st.selectbox("Select Target Audit Reference Profile Index:", options=st.session_state.flagged_idx)
            trigger_audit = st.button("🔬 EXECUTE DECONSTRUCTION", use_container_width=True)
            
        with f_col2:
            if trigger_audit:
                with st.spinner("Extracting structural impact vector paths via SHAP..."):
                    target_pos = np.where(st.session_state.flagged_idx == selected_target)[0][0]
                    X_anomaly_sample = st.session_state.proc_df.iloc[[target_pos]]
                    
                    X_train, _, y_train, _, _ = load_base_data_cache()
                    
                    # Safe check for cloud fallback mode data structure size
                    sample_size = min(100, len(X_train))
                    X_train_normal = X_train.sample(sample_size, random_state=42) if len(y_train) == 0 or int(y_train.sum()) == 0 else X_train[y_train == 0].sample(sample_size, random_state=42)
                    X_train_normal_clean = X_train_normal.drop([c for c in ['Time', 'Amount'] if c in X_train_normal.columns], axis=1, errors='ignore')
                    
                    shap_values = compute_shap_insights(st.session_state.trained_model, X_train_normal_clean, X_anomaly_sample)
                    
                    # Construct detailed premium explanations graph
                    fig_s, ax_s = plt.subplots(figsize=(10, 5))
                    fig_s.patch.set_facecolor('#070A12')
                    ax_s.set_facecolor('#0F172A')
                    
                    vals = shap_values[0]
                    features = st.session_state.proc_df.columns
                    idx_sort = np.argsort(np.abs(vals))
                    colors = ['#EF4444' if x > 0 else '#10B981' for x in vals[idx_sort]]
                    
                    ax_s.barh(range(len(idx_sort)), vals[idx_sort], color=colors, align='center', alpha=0.9)
                    ax_s.set_yticks(range(len(idx_sort)))
                    ax_s.set_yticklabels(features[idx_sort], color='#E2E8F0')
                    ax_s.set_xlabel("SHAP Impact Weight Vector Strength", color='#94A3B8')
                    ax_s.set_title(f"Risk Structural Deconstruction Chart — Incident Reference Instance #{selected_target}", color='#F8FAFC', pad=15)
                    st.pyplot(fig_s)

# --- TAB 3: PERFORMANCE LOGISTICS ---
with tab_performance:
    st.markdown("### 📊 PLATFORM BASELINE & METRIC INTEGRITY")
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color:#38BDF8; margin-top:0;'>Core Architecture Network Details</h4>
                <p style='font-size:0.9rem; color:#94A3B8;'><strong>Model Engine Type:</strong> Fully-Connected Deep Learning Residual Autoencoder</p>
                <p style='font-size:0.9rem; color:#94A3B8;'><strong>Layer Matrix Maps:</strong> Input(29) ➔ HiddenLinear(16) ➔ LatentBottleneck(8) ➔ HiddenLinear(16) ➔ Output(29)</p>
                <p style='font-size:0.9rem; color:#94A3B8;'><strong>Optimization Solvers:</strong> AdamW (Weight Decay Tuning Enabled at 1e-5)</p>
            </div>
        """, unsafe_allow_html=True)
        
    with p_col2:
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color:#A855F7; margin-top:0;'>Execution Device Information Layer</h4>
                <p style='font-size:0.9rem; color:#94A3B8;'><strong>Local Hardware Integration:</strong> Apple Silicon Metal Performance Shaders (MPS) Backend</p>
                <p style='font-size:0.9rem; color:#94A3B8;'><strong>Cloud Engine Deployment:</strong> Linux Server Environment (Standard CPU Compute Allocation)</p>
            </div>
        """, unsafe_allow_html=True)
