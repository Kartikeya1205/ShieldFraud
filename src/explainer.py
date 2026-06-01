import shap
import torch
import pandas as pd
import numpy as np

def compute_shap_insights(model, X_train_bg, X_anomalies):
    print("🔮 Phase 5: Initializing Explainable AI layer via SHAP...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    
    # Custom model wrapper that returns individual sample reconstruction error scores to SHAP
    def predict_reconstruction_error(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        with torch.no_grad():
            reconstructed = model(x_tensor)
            error = torch.mean((x_tensor - reconstructed) ** 2, dim=1)
        return error.cpu().numpy()

    # Condense background metrics into 10 reference points using K-Means for efficient execution on Mac
    bg_summary = shap.kmeans(X_train_bg.values, 10)
    explainer = shap.KernelExplainer(predict_reconstruction_error, bg_summary)
    
    # Run the Kernel explaining algorithm over the anomaly arrays
    shap_values = explainer.shap_values(X_anomalies.values)
    return shap_values

