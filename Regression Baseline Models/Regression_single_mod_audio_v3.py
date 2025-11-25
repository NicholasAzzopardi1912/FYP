import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Loading the preprocessed audio dataset
audio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/audio_data.csv")

# Splitting the dataset into features and target variables
# median_arousal and median_valence are the target variables
# Classification labels are excluded for regression
# Participant column is used for grouping in GroupKFold, hence dropped from the feature set
X = audio_df.drop(columns=["median_arousal", "median_valence", "arousal_class", "valence_class"])
y_arousal = audio_df["median_arousal"].values
y_valence = audio_df["median_valence"].values
groups = audio_df["Participant"].astype('str').values

# This will then ensure that participants do not appear in both training and testing sets
X = X.drop(columns=["Participant"]).values


# Function that Builds the regression neural network model
def build_regression_audio_model(input):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input,)), # Input layer
        Dropout(0.3), # Dropout layer for regularization (helps to prevent overfitting)
        Dense(64, activation='relu'), # Hidden layer
        Dropout(0.3), # Dropout layer for regularization
        Dense(32, activation='relu'), # Hidden layer
        Dense(1, activation='linear')  # Single output layer for regression
    ])

    # Compiling the model using Adam optimizer and Mean Squared Error loss function
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    return model

# Function to perform GroupKFold Cross-Validation
# Each fold shall train and test the model with diverse participant groups
def group_kfold_split(X, y, groups, label_name = "Arousal"):
    # Using the LOPO strategy by setting n_splits to the number of unique participants
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)

    gkf = GroupKFold(n_splits=n_splits) # Setting the splitter
    
    fold_results = {
        "fold": [], "train_mse": [], "train_pearson": [], "test_mse": [], "test_pearson": []
    }

    print(f"\nStarting GroupKFold Cross-Validation for {label_name} Model:\n")

    # Iterate over each fold, split the data based on Participant ID, build and train the model, then evaluate
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups = groups), 1):
        print(f"\nFold {fold}:")
        
        # Splitting the data into training and testing sets based on the current fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # A model is built for the current fold
        model = build_regression_audio_model(X_train.shape[1])

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        # Training the model with a validation split of 20%
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

        # Making predictions on both the train and test set
        train_predictions = model.predict(X_train, verbose=0).flatten()
        test_predictions = model.predict(X_test, verbose=0).flatten()

        # Evaluating the model's performance using MSE and PCC
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        train_pearson, _ = pearsonr(y_train, train_predictions)
        test_pearson, _ = pearsonr(y_test, test_predictions)

        # Storing the metrics for the current fold
        fold_results["fold"].append(fold)
        fold_results["train_mse"].append(train_mse)
        fold_results["train_pearson"].append(train_pearson)
        fold_results["test_mse"].append(test_mse)
        fold_results["test_pearson"].append(test_pearson)

        print(f"  Train MSE: {train_mse:.4f}, Train Pearson: {train_pearson:.4f}")
        print(f"  Test MSE: {test_mse:.4f}, Test Pearson: {test_pearson:.4f}")

    # Summarizing the performance across all folds
    print(f"\nAverage {label_name} Model Performance over {n_splits} Folds:")
    print(f" Train MSE: {np.mean(fold_results['train_mse']):.4f} ± {np.std(fold_results['train_mse']):.4f}")
    print(f" Train Pearson: {np.mean(fold_results['train_pearson']):.4f} ± {np.std(fold_results['train_pearson']):.4f}")
    print(f" Test MSE: {np.mean(fold_results['test_mse']):.4f} ± {np.std(fold_results['test_mse']):.4f}")
    print(f" Test Pearson: {np.mean(fold_results['test_pearson']):.4f} ± {np.std(fold_results['test_pearson']):.4f}")


    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f"{label_name.lower()}_regression_audio_groupkfold_results.csv", index=False)
    print(f"\nSaved fold results to {label_name.lower()}_regression_audio_groupkfold_results.csv")


    # Plotting the test metrics across folds
    plt.figure(figsize=(6, 4))
    plt.plot(results_df["fold"], results_df["test_mse"], marker='o', label='Test MSE')
    plt.plot(results_df["fold"], results_df["test_pearson"], marker='o', label='Test Pearson r')
    plt.title(f'{label_name} Regression Performance Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# Running GroupKFold Cross-Validation for Arousal and Valence models
group_kfold_split(X, y_arousal, groups, label_name="Arousal")
group_kfold_split(X, y_valence, groups, label_name="Valence")


