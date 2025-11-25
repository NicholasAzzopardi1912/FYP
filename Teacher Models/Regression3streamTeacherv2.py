# The following is the Multimodal 3-stream Regression Teacher Model

# The structure to be taken is 3 branches for 3 different modalities, where
# a seperate MLP encoder branch is used for each modality allowing modality-specific
# features to be learned.
# Using fusion, the encoders outputs are concatenated, merging the information from all modalities
# The fusion head is what learns the cross-modal interations
# The output layer is a single neuron with linear activation for regression

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from sklearn.model_selection import GroupKFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# This function will create an encoder (branch) for a given modality
# name_prefix: A string prefix to name the layers uniquely for each branch
# input_shape: The shape of the input data for this modality
# hidden_units: A tuple defining the number of units in each hidden layer
# dropout: Dropout rate to be applied after each hidden layer
# l2: L2 regularization factor helping to penalise large weights

def encoder_branch(name_prefix, input_dim, hidden_units=(128,64), dropout=0.3, l2=1e-4):
    # Input vector for the modality
    inputs = layers.Input(shape=(input_dim,), name=f"{name_prefix}_input")

    x = inputs

    for i, units in enumerate(hidden_units):
        # Dense layer with ReLU activation and L2 regularization
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2),
                         name=f"{name_prefix}_dense_{i+1}")(x)
        # Dropout layer for regularization
        x = layers.Dropout(dropout, name=f"{name_prefix}_dropout_{i+1}")(x)
    # returning the input and encoded representation of the branch
    return inputs, x


def build_teacher_regression(input_shapes, target_name='arousal', branch_hidden=(128,64), fusion_hidden=(128,32), dropout=0.3, l2 = 1e-4):
    audio_dimensions, video_dimensions, physiological_dimensions = input_shapes

    # Create encoder branches for each modality
    audio_input, audio_encoded = encoder_branch('audio', audio_dimensions, branch_hidden, dropout, l2)
    video_input, video_encoded = encoder_branch('video', video_dimensions, branch_hidden, dropout, l2)
    physio_input, physio_encoded = encoder_branch('physio', physiological_dimensions, branch_hidden, dropout, l2)

    # Fusion layer: Concatenate the encoded outputs from all branches
    fused = layers.Concatenate(name='fusion_layer')([audio_encoded, video_encoded, physio_encoded])

    # Fusion head: Further processing after fusion
    x = fused
    for i, units in enumerate(fusion_hidden):
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2),
                         name=f"fusion_dense_{i+1}")(x)
        
        x = layers.Dropout(dropout, name=f"fusion_dropout_{i+1}")(x)

    
    # Output layer for regression
    output = layers.Dense(1, activation='linear', name=f"{target_name}_output")(x)

    # Defining the full model with the three input branches and single output
    model = Model(inputs=[audio_input, video_input, physio_input], outputs=output, name=f'3stream_regression_teacher_{target_name}')

    # Compiling the model using the Adam optimizer and Mean Squared Error loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    
    return model

# Loading the csv files
audio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/audio_data.csv")
video_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/video_data.csv")
physio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/physio_data.csv")

# Paricipant label
participants = audio_df["Participant"].values
# Target variables
y_arousal = audio_df["median_arousal"].values
y_valence = audio_df["median_valence"].values
# Feature matrices for each modality
X_audio = audio_df.drop(
    columns=["Participant", "median_arousal", "median_valence", "arousal_class", "valence_class"]).values

X_video = video_df.drop(
    columns=["Participant", "median_arousal", "median_valence", "arousal_class", "valence_class"]).values

X_physio = physio_df.drop(
    columns=["Participant", "median_arousal", "median_valence", "arousal_class", "valence_class"]).values

# Function to train the teacher model with Group K-Fold Cross-Validation
def train_teacher_model(x_audio, x_video, x_physio, y, groups, target_name='arousal', epochs=200, batch_size=64, patience=15):
    # Using the LOPO strategy by setting n_splits to the number of unique participants
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    
    gkf = GroupKFold(n_splits=n_splits)
    fold = 1
    
    mse_scores, pearson_scores = [], []
    fold_results = []
    
    print(f"Starting Group K-Fold Cross-Validation Training for {target_name.upper()} ")

    for train_idx, val_idx in gkf.split(x_audio, y, groups):
        print(f"\nTraining fold {fold}/{n_splits}")

        # Splitting the data into training and validation sets for the current fold
        x_audio_train, x_audio_val = x_audio[train_idx], x_audio[val_idx]
        x_video_train, x_video_val = x_video[train_idx], x_video[val_idx]
        x_physio_train, x_physio_val = x_physio[train_idx], x_physio[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Building the model for the current fold
        input_shapes = (x_audio.shape[1], x_video.shape[1], x_physio.shape[1])
        model = build_teacher_regression(input_shapes, target_name=target_name)

        # The model is trained with early stopping based on validation loss
        model.fit(
            [x_audio_train, x_video_train, x_physio_train], y_train,
            validation_data=([x_audio_val, x_video_val, x_physio_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)],
            verbose=1)

        # Making predictions on validation set
        y_predictions = model.predict([x_audio_val, x_video_val, x_physio_val], verbose=0).flatten()

        # Computing the evaluation metrics
        mse = mean_squared_error(y_val, y_predictions)
        pearson_corr, _ = pearsonr(y_val, y_predictions)
        
        # Storing the scores for this fold
        mse_scores.append(mse)
        pearson_scores.append(pearson_corr)
        fold_results.append({'Fold': fold, 'MSE': mse, 'Pearson_Corr': pearson_corr})

        print(f" Fold {fold} - MSE: {mse:.4f}, Pearson_Corr: {pearson_corr:.4f}")
        fold += 1
        
    # Computing the mean and standard deviation of the metrics
    mean_mse, std_mse = np.mean(mse_scores), np.std(mse_scores)
    mean_pearson, std_pearson = np.mean(pearson_scores), np.std(pearson_scores)

    # Outputting the cross validation results
    print(f"\n Average Cross-Validation Results ")
    print(f"{target_name.upper()} Regression Teacher ")
    print(f"Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"Mean Pearson's: {mean_pearson:.4f} ± {std_pearson:.4f}")
    
    # Creating a DataFrame to display fold results
    fold_results.append({'Fold': 'Mean ± Std', 'MSE': f"{mean_mse:.4f} ± {std_mse:.4f}", 'Pearson_Corr': f"{mean_pearson:.4f} ± {std_pearson:.4f}"})
    
    # Saving fold results to a CSV file
    results_df = pd.DataFrame(fold_results)
    filename = f"{target_name.lower()}_teacher_regression_results.csv"
    results_df.to_csv(filename, index=False)

    print(f"\nSaved fold results to {filename}")
    
    # Plotting the performance across folds
    plt.figure(figsize=(6, 4))
    plt.plot(mse_scores, marker='o', label='MSE')
    plt.plot(pearson_scores, marker='o', label="Pearson's r")
    plt.title(f'{target_name.capitalize()} Model Performance Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# Training the teacher model for Arousal
train_teacher_model(
    x_audio=X_audio,
    x_video=X_video,
    x_physio=X_physio,
    y=y_arousal,
    groups=participants,
    target_name='arousal')

# Training the teacher model for Valence
train_teacher_model(
    x_audio=X_audio,
    x_video=X_video,
    x_physio=X_physio,
    y=y_valence,
    groups=participants,
    target_name='valence')