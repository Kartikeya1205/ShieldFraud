import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessor import load_and_preprocess
from src.autoencoder_model import train_autoencoder
from src.explainer import compute_shap_insights

# Set up page configurations for a premium, wide-screen monitor interface
st.set_page_config(page_title="ShieldFraud AI Enterprise", page_icon="🛡️", layout="wide")

# Inject High-Tech Dark / Professional Cyber-Security Styling
st.markdown("""
    <style>
    /* Main Panel Background Customizations */
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
    }
    /* Metric Card Custom Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #38BDF8 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1rem !important;
        color: #94A3B8 !important;
    }
    /* Custom High-Tech Containers */
    .crypto-card {
        background: linear-gradient(135deg, #111827 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    /* Section Headers */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-family: 'Inter', system-ui, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header Row
col_logo, col_title = st.columns([1, 11])
with col_title:
    st.title("🛡️ SHIELD-FRAUD AI • ENTERPRISE AUDIT HUB")
    st.markdown("<p style='color: #7DD3FC; margin-top: -15px;'>M2 Native Hardware-Accelerated Tabular Anomaly Detection Pipeline</p>", unsafe_allow_html=True)

st.divider()

# Sidebar Setup - Controller Dashboard
st.sidebar.markdown("### 🕹️ PIPELINE CONFIGURATION")
uploaded_file = st.sidebar.file_uploader("Ingest Transaction Stream (CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ DETECTION BOUNDARIES")
threshold_pct = st.sidebar.slider("Anomaly Sensitivity (Percentile)", min_value=90.0, max_value=99.9, value=98.5, step=0.1)

# Persistent Session Caching Initialization
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.device = None

# Heavy Background Operation Caching to maximize interface frame rates
@st.cache_data
def get_cached_base_data():
    X_train, _, y_train, _ = load_and_preprocess('data/creditcard.csv')
    X_train_normal = X_train[y_train == 0].sample(50000, random_state=42)
    return X_train, y_train, X_train_normal

# Baseline Generation Event Trigger
if st.sidebar.button("🏗️ BOOTSTRAP AUTOENCODER CORE"):
    with st.spinner("Compiling Neural Networks & Mapping Arrays directly onto Apple Silicon GPU..."):
        X_train, y_train, X_train_normal = get_cached_base_data()
        model, device = train_autoencoder(X_train_normal, epochs=8, batch_size=512)
        
        st.session_state.trained_model = model
        st.session_state.device = device
        st.sidebar.success("⚡ Unified Memory Allocation Secure!")

# Main Workstation Layout Logic
if uploaded_file is not None:
    # Safely load the stream file
    test_df = pd.read_csv(uploaded_file)
    display_df = test_df.drop('Class', axis=1, errors='ignore')
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Machine Learning Subsystems offline. Please initialize the neural model using the Control Center panel first.")
    else:
        # Preprocess uploaded batch
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        proc_df = display_df.copy()
        if 'Amount' in proc_df.columns:
            proc_df['scaled_amount'] = scaler.fit_transform(proc_df['Amount'].values.reshape(-1, 1))
            
        proc_df = proc_df.drop([c for c in ['Time', 'Amount'] if c in proc_df.columns], axis=1)
        
        # Parse inputs via Model
        model = st.session_state.trained_model
        device = st.session_state.device
        
        X_tensor = torch.tensor(proc_df.values, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            mse_losses = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            
        threshold = np.percentile(mse_losses, threshold_pct)
        flagged_indices = np.where(mse_losses > threshold)[0]
        
        # --- PANEL AREA 1: SYSTEM HEALTH STATS ---
        st.markdown("### 📊 REAL-TIME THREAT METRICS")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        kpi1.metric(label="Processed Records", value=f"{len(test_df):,}")
        
        anomaly_rate = (len(flagged_indices) / len(test_df)) * 100
        kpi2.metric(
            label="Flagged Incidents", 
            value=f"{len(flagged_indices):,}", 
            delta=f"{anomaly_rate:.2f}% Match Rate", 
            delta_color="inverse"
        )
        
        kpi3.metric(label="Decision Limit (MSE)", value=f"{threshold:.4f}")
        kpi4.metric(label="Computing Backend", value=str(device).upper())
        
        st.divider()
        
        # --- PANEL AREA 2: CHART GRID LAYOUT ---
        chart_col, table_col = st.columns([5, 3])
        
        with chart_col:
            st.markdown("### 📈 ANALYTICAL DISTRIBUTION DENSITY")
            # Apply styling directly to plot aesthetics matching the dark mode layout
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(8, 4.2))
            fig.patch.set_facecolor('#0A0E17')
            ax.set_facecolor('#111827')
            
            sns.histplot(mse_losses, bins=60, kde=True, color='#38BDF8', ax=ax, stat="density", alpha=0.6)
            ax.axvline(threshold, color='#EF4444', linestyle='--', linewidth=2.5, label=f'Alarm Threshold ({threshold:.3f})')
            
            ax.set_xlabel("Reconstruction Loss Magnitude (MSE)", color='#94A3B8')
            ax.set_ylabel("Density Profiles", color='#94A3B8')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')
            ax.legend(facecolor='#1E293B', edgecolor='#334155')
            st.pyplot(fig)
            
        with table_col:
            st.markdown("### 🚨 CRITICAL THREAT LOG (TOP 10)")
            anomaly_summary = test_df.iloc[flagged_indices].copy()
            anomaly_summary['Risk Score (MSE)'] = mse_losses[flagged_indices]
            anomaly_summary = anomaly_summary.sort_values(by='Risk Score (MSE)', ascending=False)
            
            # Interactive premium styling dataframes
            st.dataframe(
                anomaly_summary[['Risk Score (MSE)']].head(10),
                use_container_width=True,
                height=355
            )
            
        st.divider()
        
        # --- PANEL AREA 3: ADVANCED AI AUDITING LAYER ---
        st.markdown("### 🔬 EXPLAINABLE INTERPRETATION CONSOLE")
        st.markdown("Isolate individual alert indicators below to run mathematical SHAP decomposition on the underlying feature layers.")
        
        if len(flagged_indices) > 0:
            select_col, button_col = st.columns([6, 2])
            with select_col:
                selected_index = st.selectbox("Select Target Flag Index Reference Point:", options=flagged_indices)
            with button_col:
                st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
                run_shap = st.button("🔬 DECONSTRUCT RISK INDICATORS", use_container_width=True)
                
            if run_shap:
                with st.spinner("Deconstructing structural variance models..."):
                    target_pos = np.where(flagged_indices == selected_index)[0][0]
                    X_anomaly_sample = proc_df.iloc[[target_pos]]
                    
                    X_train, y_train, _ = get_cached_base_data()
                    X_train_normal = X_train[y_train == 0].sample(100, random_state=42)
                    X_train_normal_clean = X_train_normal.drop([c for c in ['Time', 'Amount'] if c in X_train_normal.columns], axis=1, errors='ignore')
                    
                    shap_values = compute_shap_insights(model, X_train_normal_clean, X_anomaly_sample)
                    
                    # Generate an advanced explanation contribution chart
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 4.5))
                    fig_shap.patch.set_facecolor('#0A0E17')
                    ax_shap.set_facecolor('#111827')
                    
                    vals = shap_values[0]
                    features = proc_df.columns
                    indices = np.argsort(np.abs(vals))
                    
                    # Style horizontal bars dynamically based on directional impact direction
                    bar_colors = ['#EF4444' if x > 0 else '#10B981' for x in vals[indices]]
                    
                    ax_shap.barh(range(len(indices)), vals[indices], color=bar_colors, align='center', alpha=0.85)
                    ax_shap.set_yticks(range(len(indices)))
                    ax_shap.set_yticklabels(features[indices], color='#E2E8F0')
                    ax_shap.tick_params(colors='#94A3B8')
                    ax_shap.set_xlabel("SHAP Value Impact Vector Strength", color='#94A3B8')
                    ax_shap.set_title(f"Risk Profile Deconstruction — Transaction ID Account Reference #{selected_index}", color='#F8FAFC', pad=15)
                    ax_shap.spines['top'].set_visible(False)
                    ax_shap.spines['right'].set_visible(False)
                    ax_shap.spines['left'].set_color('#334155')
                    ax_shap.spines['bottom'].set_color('#334155')
                    
                    st.pyplot(fig_shap)
                    st.info("💡 Interpretive Guide: Crimson indicators highlight structural variables driving the transaction over the dynamic cutoff barrier; green indicators indicate stabilizing patterns.")
        else:
            st.success("🎉 Batch Verification Clean. No structural variance markers discovered above threshold boundaries.")

else:
    # Premium Welcome Splash View
    st.markdown("""
        <div style='background-color: #111827; border: 1px solid #334155; padding: 40px; border-radius: 12px; text-align: center; margin-top: 50px;'>
            <h2 style='color: #38BDF8;'>Ready for Batch Ingestion</h2>
            <p style='color: #94A3B8; max-width: 600px; margin: 0 auto; padding-top: 10px;'>
                Please use the Pipeline Control Center on the left panel to upload your target credit card ledger transaction files (.csv). 
                Ensure the platform core has been initialized to connect your data paths with the physical GPU backend.
            </p>
        </div>
    """, unsafe_allow_html=True)
