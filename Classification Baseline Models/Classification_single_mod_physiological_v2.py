import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
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
    fold_acc = []

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
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

        # Making predictions on the test set 
        predictions_probability = model.predict(X_test, verbose=0)
        y_predict = (predictions_probability > 0.5).astype(int).flatten()


        # Storing the metrics for the current fold
        accuracy = accuracy_score(y_test, y_predict)
        fold_acc.append(accuracy)

        # Displaying classification report for the current fold
        print(f" Fold {fold} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_predict, digits=3))


    # Printing the overall mean accuracy across all folds
    print(f"\n Average {label_name} Classification Performance across {n_splits} folds:")
    print(f" Accuracy: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")


    # Plotting the performance metrics across folds
    plt.figure(figsize=(6, 4))
    plt.plot(fold_acc, marker ='o', label = 'Accuracy')
    plt.title(f'{label_name} Classification Accuracy across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Running GroupKFold Cross-Validation for Arousal and Valence models
group_kfold_split(X, y_arousal, groups, n_splits = 5, label_name="Arousal")
group_kfold_split(X, y_valence, groups, n_splits = 5, label_name="Valence")