import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Loading the preprocessed physiological dataset
physio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/physio_data.csv")

# Splitting the dataset into features and target variables
# Removing continous and classification target columns from the feature set
# Participant column is used for grouping in GroupKFold, hence dropped from the feature set
X = physio_df.drop(columns=["median_arousal", "median_valence", "arousal_class", "valence_class"])
y_arousal = physio_df["arousal_class"].values
y_valence = physio_df["valence_class"].values
groups = physio_df["Participant"].astype('str').values

# This will then ensure that participants do not appear in both training and testing sets
X = X.drop(columns=["Participant"]).values

# Function that builds the classification neural network model
def build_classification_physio_model(input):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input,)), # Input layer
        Dropout(0.3), # Dropout layer for regularization (helps to prevent overfitting)
        Dense(64, activation='relu'), # Hidden layer
        Dropout(0.3), # Dropout layer for regularization
        Dense(32, activation='relu'), # Hidden layer
        Dense(1, activation='sigmoid')  # Single output layer for classification
    ])

    # Compiling the model using Adam optimizer and binary cross-entropy loss function
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Function to perform GroupKFold Cross-Validation
# Each fold shall train and test the model with diverse participant groups
def group_kfold_split(X, y, groups, n_splits = 5, label_name = "Arousal"):
    gkf = GroupKFold(n_splits=n_splits) # Setting the splitter
    
    fold_results = {
        "fold": [], "train_accuracy": [], "test_accuracy": [],
        "train_f1": [], "test_f1": [], "train_pearson": [], "test_pearson": []
    }

    print(f"\nStarting GroupKFold Cross-Validation for {label_name} Model:\n")
    # Iterate over each fold, split the data based on Participant ID, build and train the model, then evaluate
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups = groups), 1):
        print(f"\nFold {fold}:")
        
        # Splitting the data into training and testing sets based on the current fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # A model is built for the current fold
        model = build_classification_physio_model(X_train.shape[1])

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        # Training the model with a validation split of 20%
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

        # Making probabilistic predictions on both the train and test set 
        train_probability = model.predict(X_train, verbose=0).flatten()
        test_probability = model.predict(X_test, verbose=0).flatten()

        # Converting probabilities to class labels using a threshold of 0.5
        train_predictions = (train_probability > 0.5).astype(int)
        test_predictions = (test_probability > 0.5).astype(int)


        # Calculating the metrics for the current fold
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        train_f1 = f1_score(y_train, train_predictions)
        test_f1 = f1_score(y_test, test_predictions)
        train_pearson, _ = pearsonr(y_train, train_probability)
        test_pearson, _ = pearsonr(y_test, test_probability)

        # Storing the results of the metrics for the current fold
        fold_results["fold"].append(fold)
        fold_results["train_accuracy"].append(train_accuracy)
        fold_results["test_accuracy"].append(test_accuracy)
        fold_results["train_f1"].append(train_f1)
        fold_results["test_f1"].append(test_f1)
        fold_results["train_pearson"].append(train_pearson)
        fold_results["test_pearson"].append(test_pearson)

        print(f" Train Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}, Pearson Correlation: {train_pearson:.4f}")
        print(f" Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}, Pearson Correlation: {test_pearson:.4f}")


    # Summarizing the average performance across all folds
    print(f"\n Average {label_name} Model Performance across {n_splits} Folds:")
    print(f" Train Accuracy: {np.mean(fold_results['train_accuracy']):.4f} ± {np.std(fold_results['train_accuracy']):.4f}")
    print(f" Test Accuracy: {np.mean(fold_results['test_accuracy']):.4f} ± {np.std(fold_results['test_accuracy']):.4f}")
    print(f" Train F1 Score: {np.mean(fold_results['train_f1']):.4f} ± {np.std(fold_results['train_f1']):.4f}")
    print(f" Test F1 Score: {np.mean(fold_results['test_f1']):.4f} ± {np.std(fold_results['test_f1']):.4f}")
    print(f" Train Pearson Correlation: {np.mean(fold_results['train_pearson']):.4f} ± {np.std(fold_results['train_pearson']):.4f}")
    print(f" Test Pearson Correlation: {np.mean(fold_results['test_pearson']):.4f} ± {np.std(fold_results['test_pearson']):.4f}")


    # Saving the fold results to a CSV file
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f"{label_name.lower()}_classification_physio_groupkfold_results.csv", index=False)
    print(f"\nSaved fold results to {label_name.lower()}_classification_physio_groupkfold_results.csv")

    # Plotting the test metrics across folds
    plt.figure(figsize=(6, 4))
    plt.plot(results_df["fold"], results_df["test_accuracy"], marker='o', label='Test Accuracy')
    plt.plot(results_df["fold"], results_df["test_f1"], marker='o', label='Test F1')
    plt.plot(results_df["fold"], results_df["test_pearson"], marker='o', label='Test Pearson r')
    plt.title(f'{label_name} Classification Performance Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# Running GroupKFold Cross-Validation for Arousal and Valence models
group_kfold_split(X, y_arousal, groups, n_splits = 5, label_name="Arousal")
group_kfold_split(X, y_valence, groups, n_splits = 5, label_name="Valence")