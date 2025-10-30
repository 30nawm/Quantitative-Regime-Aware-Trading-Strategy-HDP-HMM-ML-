import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

def prepare_data_for_modeling(features_df, labels_series, test_split_size):
    
    combined = pd.concat([features_df, labels_series.rename('target')], axis=1)
    combined.dropna(inplace=True)
    
    X = combined.drop('target', axis=1)
    y = combined['target']
    
    # Remap y from [-1, 0, 1] to [0, 1, 2] for multiclass models
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})
    
    split_index = int(len(X) * (1 - test_split_size))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y_mapped.iloc[:split_index]
    y_test = y_mapped.iloc[split_index:]
    
    # Scale data (especially important for NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def train_xgboost(X_train, y_train):
    print("Training XGBoost model...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        objective='multi:softmax',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost training complete.")
    return xgb_model

def create_nn_model(input_dim):
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax') # 3 classes: Sell (-1), Hold (0), Buy (1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_neural_network(X_train_scaled, y_train, epochs, batch_size):
    print("Training Neural Network model...")
    input_dim = X_train_scaled.shape[1]
    nn_model = create_nn_model(input_dim)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = nn_model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    print("Neural Network training complete.")
    return nn_model