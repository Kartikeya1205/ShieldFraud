import pandas as pd
from ctgan import CTGAN
import torch

def balance_data(X_train, y_train, method='ctgan'):
    # Detect Apple Silicon backend
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    if method == 'ctgan':
        print(f"🧬 Phase 2: Training Tabular Generative Adversarial Network (CTGAN)...")
        
        # Isolate the minor fraud class for training
        minority_data = X_train[y_train == 1].copy()
        minority_data['Class'] = 1
        
        # FIX: Removed 'device=device' from the initializer to prevent the TypeError
        ctgan = CTGAN(epochs=20, batch_size=500)
        ctgan.fit(minority_data, discrete_columns=['Class'])
        
        # Synthesize copies to inject clean statistical data variance
        num_samples_needed = 2000  
        print(f"✨ Synthesizing {num_samples_needed} brand-new structural fraud vectors...")
        synthetic_data = ctgan.sample(num_samples_needed)
        
        X_synth = synthetic_data.drop('Class', axis=1)
        y_synth = synthetic_data['Class']
        
        X_res = pd.concat([X_train, X_synth], axis=0).reset_index(drop=True)
        y_res = pd.concat([y_train, y_synth], axis=0).reset_index(drop=True)
        return X_res, y_res
    else:
        print("⚠️ Skipping data balancing. Using baseline distributions.")
        return X_train, y_train
