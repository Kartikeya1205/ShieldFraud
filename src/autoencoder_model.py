import torch
import torch.nn as nn
import numpy as np

class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(FraudAutoencoder, self).__init__()
        # Encoder Network: Compresses 29 dimensional feature arrays down to 8 dimensional bottlenecks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        # Decoder Network: Translates bottleneck features back to original configurations
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(X_train_normal, epochs=10, batch_size=512):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    input_dim = X_train_normal.shape[1]
    model = FraudAutoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    
    # Pack array elements into PyTorch DataLoader streams
    X_tensor = torch.tensor(X_train_normal.values, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("📉 Phase 3: Optimizing reconstruction weights on M2 GPU...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   Epoch [{epoch+1}/{epochs}] | Mean Squared Error Loss: {total_loss/len(loader):.6f}")
            
    return model, device
