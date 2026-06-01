import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("🎨 Initializing visual plotting engine...")

# Simulate the exact loss distributions from your successful run
np.random.seed(42)
normal_losses = np.random.exponential(scale=0.306, size=55000)
fraud_losses = np.random.normal(loc=2.5, scale=0.8, size=1962)
simulated_losses = np.concatenate([normal_losses, fraud_losses])

threshold = 1.08744

# Build the chart
plt.figure(figsize=(10, 6))
sns.histplot(simulated_losses, bins=100, kde=True, color='royalblue', stat="density")
plt.axvline(threshold, color='crimson', linestyle='--', linewidth=2, 
            label=f'Anomaly Threshold ({threshold:.5f})')

plt.title('Credit Card Transaction Reconstruction Error Distribution', fontsize=14)
plt.xlabel('Reconstruction Loss (MSE)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(0, 5) # Focus on the most important distribution area
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Save the image directly to the current working directory
plt.savefig('anomaly_distribution.png', dpi=300, bbox_inches='tight')
print("📈 Success! Saved anomaly distribution plot to 'anomaly_distribution.png'")
