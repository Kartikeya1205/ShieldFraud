import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def load_and_preprocess(data_path):
    print("📋 Phase 1: Loading raw dataset and initializing preprocessing...")
    
    # Load the 284,807 transactions
    df = pd.read_csv(data_path)
    
    # Scale 'Amount' using RobustScaler (uses median & IQR) to handle extreme transaction values safely
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Drop raw Time and Amount features to avoid leaking unneeded structural trends
    df = df.drop(['Time', 'Amount'], axis=1)
    
    # Separate features (X) from the target fraud label (y)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Stratified split ensures that the test set gets the exact same 0.17% fraud ratio as the training set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Dataset split complete. Training size: {len(X_train)}, Testing size: {len(X_test)}")
    return X_train, X_test, y_train, y_test
