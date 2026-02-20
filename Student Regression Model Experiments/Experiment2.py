# Experiment 2: reducing dominance over MSE and helping the student still learn labels.

# Importing libraries
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping
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
    # Student model input -> audio features only
    audio_input = layers.Input(shape=(input_dimensions,), name="audio_student_input")

    # First hidden layer for audio regression
    x = layers.Dense(128, activation="relu", name="dense_128")(audio_input)
    # Dropout for regularization
    x = layers.Dropout(0.3, name="drop_1")(x)

    # Second hidden layer for audio regression
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    # Dropout for regularization
    x = layers.Dropout(0.3, name="drop_2")(x)

    # Bottleneck representation used for LUPI alignment with teacher
    rep = layers.Dense(32, activation='relu', name="student_representation")(x)

    # Final Prediction
    outputs = layers.Dense(1, activation='linear', name="regression_output")(rep)

    # Defining the audio-only student regression model
    model = Model(inputs=audio_input, outputs=outputs, name="audio_student_regression")
    
    return model

def cosine_distance(T, S, epsilon=1e-8):
    # L2 normalise both representations to compute cosine similarity
    T_normalised = tf.nn.l2_normalize(T, axis=1)
    S_normalised = tf.nn.l2_normalize(S, axis=1)

    # Cosine similarity is the dot product of the normalised vectors
    cosine_similarity = tf.reduce_sum(T_normalised * S_normalised, axis=1)

    # Converting similarity to distance (1 - similarity)
    cosine_distance = 1.0 - cosine_similarity
    # Mean distance across the batch
    return tf.reduce_mean(cosine_distance)


class LUPIStudentRegressor(tf.keras.Model):
    def __init__(self, student_model, teacher_representation_model, alpha=0.5, name="lupi_student_regressor"):
        # Wrapper model that trains a student with optional teacher representation loss
        super().__init__(name=name)
        self.student_model = student_model
        self.teacher_representation_model = teacher_representation_model
        # Alpha controls the bias between task loss and representation loss
        self.alpha = float(alpha)

        # Sub-model to extract the student's bottleneck representation
        self.student_rep_model = tf.keras.Model(
            inputs=self.student_model.input,
            outputs=self.student_model.get_layer("student_representation").output)

        # Trackers for reporting losses during training/evaluation
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse_loss")
        self.rep_tracker = tf.keras.metrics.Mean(name="representation_loss")
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.mse_tracker, self.rep_tracker]
    
    def call(self, audio_x, training=False):
        # Forward pass uses the underlying student model
        return self.student_model(audio_x, training=training)
    
    def train_step(self, data):
        # Unpacking the data, where x contains all modalities and y is the target
        x, y = data

        # Audio is being used by the student, while video and physio are only for the teacher's representation loss
        audio_x = x["audio"]
        video_x = x["video"]
        physio_x = x["physio"]

        with tf.GradientTape() as tape:
            # Student prediction from audio-only input
            y_hat = self.student_model(audio_x, training=True)

            # Task loss: Mean Squared Error between student predictions and true targets
            mse_loss = tf.reduce_mean(tf.keras.losses.mse(y, y_hat))

            if self.alpha == 0.0:
                # If alpha is 0, we only care about the task loss and ignore the representation loss
                rep_loss = tf.constant(0.0, dtype=tf.float32)
                
                # Total loss is just the MSE loss when alpha is 0 (no LUPI effect)
                total_loss = mse_loss
            else:
                # Teacher representation from all modalities
                T = self.teacher_representation_model([audio_x, video_x, physio_x], training=False)
                # Student representation from audio-only input
                S = self.student_rep_model(audio_x, training=True)

                # Representation alignment loss: Cosine distance between teacher and student representations
                rep_loss = cosine_distance(T, S)
                # Squaring the representation loss to penalise larger distances more heavily
                rep_loss = tf.square(rep_loss)
                # Weighted combination of task loss and representation loss based on alpha
                total_loss = (1.0 - self.alpha) * mse_loss + self.alpha * rep_loss
        
        # Backpropagation only on the student model's trainable variables, as the teacher is fixed
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Updating trackers
        self.loss_tracker.update_state(total_loss)
        self.mse_tracker.update_state(mse_loss)
        self.rep_tracker.update_state(rep_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Evaluation step using audio-only input
        x, y = data

        # Supporting both dictionary inputs or raw arrays
        if isinstance(x, dict):
            audio_x = x["audio"]
        else:
            audio_x = x
        
        # Student prediction
        y_hat = self.student_model(audio_x, training=False)
        # Regression Loss
        mse_loss = tf.reduce_mean(tf.keras.losses.mse(y, y_hat))

        # Representation loss not computed at test time
        rep_loss = tf.constant(0.0, dtype=tf.float32)
        total_loss = mse_loss

        # Updating trackers
        self.loss_tracker.update_state(total_loss)
        self.mse_tracker.update_state(mse_loss)
        self.rep_tracker.update_state(rep_loss)

        return {m.name: m.result() for m in self.metrics}
    

# Loading the csv files
audio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/audio_data.csv")
video_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/video_data.csv")
physio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/physio_data.csv")

# Participant label
groups = audio_df["Participant"].astype(str).values

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


def train_student_regressor(X_audio, X_video, X_physio, y, groups, target_name = "arousal", alpha=0.0, student_epochs=50, student_batch_size=32, teacher_epochs=200, teacher_batch_size=64, teacher_patience=15, student_patience=8):
    # LOPO protocol using GroupKFold, where each fold corresponds to leaving one participant out for testing
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)

    # Storing test results for each fold to compute mean and std at the end
    test_mse_scores = []
    test_pearson_scores = []

    print(f"\nTraining student regressor for target: {target_name} with alpha: {alpha}")
    print(f"LOPO folds: {n_splits}")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_audio, y, groups=groups), start=1):
        print(f"\nFold {fold}/{n_splits}")

        # Splitting audio data and targets into training and testing sets based on the current fold's indices
        X_audio_train, X_audio_test = X_audio[train_idx], X_audio[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Early stopping for student training to prevent overfitting, monitoring validation loss
        early_stopping_student = EarlyStopping(monitor="val_loss", patience=student_patience, restore_best_weights=True)

        if alpha == 0.0:
            # Training audio-only student with standard supervised loss
            student = build_audio_student_regression_model(X_audio_train.shape[1])
            student.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

            student.fit(
                X_audio_train, y_train,
                validation_split=0.2,
                epochs=student_epochs,
                batch_size=student_batch_size,
                callbacks=[early_stopping_student],
                verbose=0)
            
            # Prediction on held-out participant
            y_pred = student.predict(X_audio_test, verbose=0).flatten()

        else:
            # Additional modalities available for teacher training
            X_video_train= X_video[train_idx]
            X_physio_train= X_physio[train_idx]

            # Building and training the multimodal teacher on this fold
            input_shapes = (X_audio_train.shape[1], X_video_train.shape[1], X_physio_train.shape[1])
            teacher_model = build_teacher_regression(input_shapes, target_name=target_name)

            # Early stopping for teacher to avoid overfitting
            early_stopping_teacher = EarlyStopping(monitor="val_loss", patience=teacher_patience, restore_best_weights=True)

            teacher_model.fit(
                [X_audio_train, X_video_train, X_physio_train],
                y_train,
                validation_split=0.2,
                epochs=teacher_epochs,
                batch_size=teacher_batch_size,
                callbacks=[early_stopping_teacher],
                verbose=0)

            # Extracting the teacher fusion representation to act as privileged information for the student
            teacher_representation_model = tf.keras.Model(
                inputs=teacher_model.inputs,
                outputs=teacher_model.get_layer("fusion_dense_2").output,
                name ="teacher_representation_model")
            # Freezing the teacher representation model during student training
            teacher_representation_model.trainable = False

            # Build student and wrap with LUPI training logic
            student = build_audio_student_regression_model(X_audio_train.shape[1])
            lupi_model = LUPIStudentRegressor(student, teacher_representation_model, alpha=alpha)
            lupi_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

            # Provide all modalities as input during training
            x_train_dict = {"audio": X_audio_train, "video": X_video_train, "physio": X_physio_train}

            lupi_model.fit(
                x_train_dict,
                y_train,
                validation_split=0.2,
                epochs=student_epochs,
                batch_size=student_batch_size,
                callbacks=[early_stopping_student],
                verbose=0)

            # Prediction using audio-only student path at test time
            y_pred = lupi_model.predict(X_audio_test, verbose=0).flatten()

        # Evaluating the student's performance on the held-out participant using MSE and Pearson correlation
        mse = mean_squared_error(y_test, y_pred)
        r, _ = pearsonr(y_test, y_pred)
        # Handling constant targets causing NaN Pearson r
        if np.isnan(r):
            r = np.nan

        test_mse_scores.append(mse)
        test_pearson_scores.append(r)

        print(f"Test MSE: {mse:.6f}, Test Pearson r: {r:.6f}")

    # Aggregating metrics across folds
    mean_mse = float(np.mean(test_mse_scores))
    std_mse = float(np.std(test_mse_scores))

    mean_pearson = float(np.nanmean(test_pearson_scores))
    std_pearson = float(np.nanstd(test_pearson_scores))
    valid_pearson = int(np.sum(~np.isnan(test_pearson_scores)))

    print(f"\nFinal Results for target: {target_name} with alpha: {alpha}")
    print(f"Test MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"Test Pearson (valid folds {valid_pearson}/{n_splits}): {mean_pearson:.6f} ± {std_pearson:.6f}")

    # Saving  results to a CSV file
    results_df = pd.DataFrame({
        "Fold": list(range(1, n_splits + 1)),
        "Test_MSE": test_mse_scores,
        "Test_Pearson": test_pearson_scores})

    results_df.loc[n_splits] = ["Mean ± Std", f"{mean_mse:.6f} ± {std_mse:.6f}", f"{mean_pearson:.6f} ± {std_pearson:.6f}"]

    # Exporting results to CSV
    out_csv = f"audio_student_regressor_EXP2_{target_name}_alpha_{alpha}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # Plotting fold-wise MSE to visualise performance stability across participants
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, n_splits + 1), test_mse_scores, marker="o")
    plt.title(f"Student ({target_name}) - alpha={alpha}")
    plt.xlabel("Fold")
    plt.ylabel("Test MSE")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# Running the training for both arousal and valence targets across different alpha values to compare the effect of LUPI on student performance
for alpha in [0.25, 0.5, 0.75, 1.0]:
    for target_name, y in {"arousal": y_arousal, "valence": y_valence}.items():
        train_student_regressor(
            X_audio=X_audio, X_video=X_video, X_physio=X_physio,
            y=y, groups=groups,
            target_name=target_name,
            alpha=alpha,
            student_epochs=50,
            student_batch_size=32)

