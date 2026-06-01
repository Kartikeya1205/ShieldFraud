import numpy as np
import torch
import pandas as pd
from src.data_preprocessor import load_and_preprocess
from src.dynamic_balancer import balance_data
from src.autoencoder_model import train_autoencoder
from src.explainer import compute_shap_insights

def main():
    print("==========================================================")
    print("🛡️ RUNNING END-TO-END CREDIT CARD ANOMALY PIPELINE")
    print("==========================================================")
    
    # Execute Pipeline
    X_train, X_test, y_train, y_test = load_and_preprocess('data/creditcard.csv')
    X_res, y_res = balance_data(X_train, y_train, method='ctgan')
    
    # The Autoencoder only learns structural characteristics of standard transactions
    X_train_normal = X_res[y_res == 0]
    
    # Model Training Execution
    model, device = train_autoencoder(X_train_normal, epochs=10, batch_size=512)
    
    # Evaluate anomaly thresholds on test dataset
    print("\n🔍 Phase 4: Scoring out-of-sample data points...")
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_test_tensor)
        mse_losses = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
    # Dynamically establish threshold flag boundaries at 98th percentile
    threshold = np.percentile(mse_losses, 98)
    flagged_indices = np.where(mse_losses > threshold)[0]
    
    print(f"🎯 Dynamic Anomaly Cut-off Value: {threshold:.5f}")
    print(f"🚨 Flagged {len(flagged_indices)} higher risk anomalies out of {len(X_test)} validation records.")
    
    # Run SHAP over top 2 anomalies to provide actionable insights for fraud auditors
    top_anomalies_idx = flagged_indices[np.argsort(mse_losses[flagged_indices])[-2:]]
    X_anomalies = X_test.iloc[top_anomalies_idx]
    
    shap_values = compute_shap_insights(model, X_train_normal.sample(100, random_state=42), X_anomalies)
    
    print("\n👑 Audit Completed successfully! Project pipeline is fully operational.")
    print("==========================================================")

if __name__ == "__main__":
    main()
