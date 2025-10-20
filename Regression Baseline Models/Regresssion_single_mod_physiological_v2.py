import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Loading the preprocessed physiological dataset
physio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/physio_data.csv")

# Splitting the dataset into features and target variables
X = physio_df.drop(columns=["median_arousal", "median_valence", "arousal_class", "valence_class"])
y_arousal = physio_df["median_arousal"].values
y_valence = physio_df["median_valence"].values
groups = physio_df["Participant"].astype('str').values

X = X.drop(columns=["Participant"]).values


# Building the neural network model
def build_regression_physiological_model(input):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input,)), # Input layer
        Dropout(0.3), # Dropout layer for regularization
        Dense(64, activation='relu'), # Hidden layer
        Dropout(0.3), # Dropout layer for regularization
        Dense(32, activation='relu'), # Hidden layer
        Dense(1, activation='linear')  # Single output layer for regression
    ])

    # Compiling the model using Adam optimizer and Mean Squared Error loss function
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    return model

def group_kfold_split(X, y, groups, n_splits = 5, label_name = "Arousal"):
    gkf = GroupKFold(n_splits=n_splits)
    fold_mse, fold_mae, fold_r2 = [], [], []

    print(f"\nStarting GroupKFold Cross-Validation for {label_name} Model:\n")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups = groups), 1):
        print(f"\nFold {fold}:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_regression_physiological_model(X_train.shape[1])

        early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

        predictions = model.predict(X_test, verbose=0)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        fold_mse.append(mse)
        fold_mae.append(mae)
        fold_r2.append(r2)

        print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}\n")

    print(f"Average {label_name} Model Performance over {n_splits} Folds:")
    print(f"  MSE: {np.mean(fold_mse):.4f} ± {np.std(fold_mse):.4f}")
    print(f"  MAE: {np.mean(fold_mae):.4f} ± {np.std(fold_mae):.4f}")
    print(f"  R2: {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}\n")


    plt.figure(figsize=(6, 4))
    plt.plot(fold_mse, marker ='o', label = 'MSE')
    plt.plot(fold_mae, marker ='o', label = 'MAE')
    plt.title(f'{label_name} Model Performance Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

group_kfold_split(X, y_arousal, groups, n_splits = 5, label_name="Arousal")
group_kfold_split(X, y_valence, groups, n_splits = 5, label_name="Valence")


