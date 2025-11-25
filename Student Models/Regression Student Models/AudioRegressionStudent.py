import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, regularizers
from sklearn.model_selection import GroupKFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

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

def build_audio_student_regression_model(input_dimensions):
    inputs = layers.Input(shape=(input_dimensions,), name="audio_student_input")

    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    rep = layers.Dense(32, activation='relu', name="student_representation")(x)

    outputs = layers.Dense(1, activation='linear')(rep)

    model = Model(inputs=inputs, outputs=outputs, name="audio_student_regression")

    # Compile with optimizer only; loss is handled manually in the training loop
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    return model

def cosine_distance(T, S, epsilon=1e-8):
    T_normalised = tf.nn.l2_normalize(T, axis=1)
    S_normalised = tf.nn.l2_normalize(S, axis=1)

    cosine_similarity = tf.reduce_sum(T_normalised * S_normalised, axis=1)

    cosine_distance = 1.0 - cosine_similarity
    return tf.reduce_mean(cosine_distance)

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


def train_audio_regression_student_with_teacher(X_audio, X_video, X_physio, y, groups, alpha=0.5, target_name="arousal", teacher_epochs=200, student_epochs=50, batch_size=64, teacher_patience=15):
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)

    gkf = GroupKFold(n_splits=n_splits)

    fold = 1
    test_mse_scores = []
    test_pearson_scores = []

    print(f"\n Audio Regression Student Model Training with Teacher for {target_name.upper()}")
    print(f"Teacher Influence (alpha): {alpha}")
    print(f"LOPO with {n_splits} participants/folds\n")

    for train_idx, test_idx in gkf.split(X_audio, y, groups):
        print(f"Fold {fold}/{n_splits}")

        X_audio_train, X_audio_test = X_audio[train_idx], X_audio[test_idx]
        X_video_train, X_video_test = X_video[train_idx], X_video[test_idx]
        X_physio_train, X_physio_test = X_physio[train_idx], X_physio[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        input_shapes = (X_audio_train.shape[1], X_video_train.shape[1], X_physio_train.shape[1])
        teacher = build_teacher_regression(input_shapes, target_name=target_name)

        early_stopping_teacher = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=teacher_patience, restore_best_weights=True)
        
        teacher.fit(
            [X_audio_train, X_video_train, X_physio_train],
            y_train,
            validation_split=0.2,
            epochs=teacher_epochs,
            batch_size=batch_size,
            callbacks=[early_stopping_teacher],
            verbose=0
        )

        teacher_representation_layer = teacher.get_layer('fusion_dense_2')
        teacher_representation_model = tf.keras.Model(
            inputs=teacher.inputs, outputs=teacher_representation_layer.output)
        
        student = build_audio_student_regression_model(X_audio_train.shape[1])

        student_representation_layer = student.get_layer("student_representation")
        student_representation_model = tf.keras.Model(
            inputs=student.input, outputs=student_representation_layer.output)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        train_dataset = tf.data.Dataset.from_tensor_slices(((X_audio_train, X_video_train, X_physio_train), y_train))

        train_dataset = train_dataset.shuffle(buffer_size=len(X_audio_train)).batch(batch_size)

        for epoch in range(1, student_epochs +1):
            epoch_losses = []

            for (X_audio_batch, X_video_batch, X_physio_batch), y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    T = teacher_representation_model([X_audio_batch, X_video_batch, X_physio_batch], training=False)

                    y_prediction = student(X_audio_batch, training=True)
                    S = student_representation_model(X_audio_batch, training=True)

                    data_loss = tf.reduce_mean(tf.keras.losses.mse(y_batch, y_prediction))

                    representation_loss = cosine_distance(T, S)

                    total_loss = (1-alpha) * data_loss + alpha * representation_loss

                gradients = tape.gradient(total_loss, student.trainable_variables)
                optimizer.apply_gradients(zip(gradients, student.trainable_variables))

                epoch_losses.append(total_loss.numpy())
            
            average_epoch_loss = np.mean(epoch_losses)
            if epoch % 10 == 0 or epoch == 1:
                print(f" Epoch {epoch}/{student_epochs} - Total Loss: {average_epoch_loss:.4f}")
        
        y_prediction_test = student.predict(X_audio_test, verbose=0).flatten()

        mse = mean_squared_error(y_test, y_prediction_test)
        pearson_corr, _ = pearsonr(y_test, y_prediction_test)

        test_mse_scores.append(mse)
        test_pearson_scores.append(pearson_corr)

        print(f"Fold {fold} - Test MSE: {mse:.4f}, Test Pearson: {pearson_corr:.4f}")

        fold +=1

    mean_mse = np.mean(test_mse_scores)
    std_mse = np.std(test_mse_scores)
    mean_pearson = np.mean(test_pearson_scores)
    std_pearson = np.std(test_pearson_scores)

    print("\n Final Results for Audio Regression Student Model")
    print(f"Target: {target_name.upper()}, alpha = {alpha}")
    print(f"Test MSE:   {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"Test Pearson:   {mean_pearson:.4f} ± {std_pearson:.4f}")

    results_df = pd.DataFrame({
        "Fold": list(range(1, n_splits + 1)),
        "Test_MSE": test_mse_scores,
        "Test_Pearson": test_pearson_scores
    })
    
    results_df.loc[n_splits] = ["Mean ± Std", f"{mean_mse:.4f} ± {std_mse:.4f}", f"{mean_pearson:.4f} ± {std_pearson:.4f}"]


    filename = f"audio_{target_name.lower()}_student_regression_alpha_{alpha}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\n Saved student model results to {filename}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, n_splits + 1), test_mse_scores, marker='o', label='Test MSE')
    plt.title(f'Audio {target_name.capitalize()} Student model, alpha={alpha}')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


targets = {"arousal": y_arousal, "valence": y_valence}

alphas = [0.25, 0.5, 0.75, 1.0]

for target_name, y in targets.items():
    for alpha in alphas:
        train_audio_regression_student_with_teacher(
            X_audio=X_audio,
            X_video=X_video,
            X_physio=X_physio,
            y=y,
            groups=participants,
            alpha=alpha,
            target_name=target_name)