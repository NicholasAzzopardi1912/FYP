# Classification Student Model (Prediction Distillation with KL Divergence)

# Importing Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr
import os
import json

RUN_DIR = "runs_audio_student_classification_kl"
os.makedirs(RUN_DIR, exist_ok=True)

# Shared encoder branch for each modality in the teacher model
def encoder_branch(name_prefix, input_dim, hidden_units=(128, 64), dropout=0.3, l2=1e-4):
    # Input vector for the modality
    inputs = layers.Input(shape=(input_dim,), name=f"{name_prefix}_input")

    x = inputs

    for i, units in enumerate(hidden_units):
        # Dense layer with ReLU activation and L2 regularization
        x = layers.Dense(units, activation="relu",
                        kernel_regularizer=regularizers.l2(l2),
                        name=f"{name_prefix}_dense_{i+1}")(x)
        
        # Dropout for regularization
        x = layers.Dropout(dropout, name=f"{name_prefix}_dropout_{i+1}")(x)
    # Returning the input and encoded representation
    return inputs, x

def build_teacher_classification(input_shapes, target_name="arousal", branch_hidden=(128, 64), fusion_hidden=(128, 32), dropout=0.3, l2=1e-4):
    # Unpacking input shapes for each modality
    audio_dimensions, video_dimensions, physiological_dimensions = input_shapes

    # Creating separate encoder branches for each modality
    audio_input, audio_encoded = encoder_branch("audio", audio_dimensions, branch_hidden, dropout, l2)
    video_input, video_encoded = encoder_branch("video", video_dimensions, branch_hidden, dropout, l2)
    physio_input, physio_encoded = encoder_branch("physio", physiological_dimensions, branch_hidden, dropout, l2)

    # Fusion Layer: Concatenating encoded outputs from all branches
    fused = layers.Concatenate(name="fusion_layer")([audio_encoded, video_encoded, physio_encoded])

    # Fusion head to apply further processing after fusion
    x = fused
    for i, units in enumerate(fusion_hidden):
        x = layers.Dense(units, activation="relu",
                        kernel_regularizer=regularizers.l2(l2),
                        name=f"fusion_dense_{i+1}")(x)
        
        x = layers.Dropout(dropout, name=f"fusion_dropout_{i+1}")(x)

    # Output layer for binary classification (sigmoid probability)
    output = layers.Dense(1, activation="sigmoid", name=f"{target_name}_output")(x)

    # Defining the full teacher model with three input branches and single output
    model = Model(inputs=[audio_input, video_input, physio_input], outputs=output, name=f"3stream_classification_teacher_{target_name}")
    
    # Compiling the model using Adam optimizer and binary cross-entropy loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

def build_audio_student_classification_model(input_dimensions):
    # Input layer for audio features
    audio_input = layers.Input(shape=(input_dimensions,), name="audio_student_input")

    # First hidden layer for audio classification
    x = layers.Dense(128, activation="relu", name="dense_128")(audio_input)
    # Dropout for regularization
    x = layers.Dropout(0.3, name="dropout_1")(x)

    # Second hidden layer for audio classification
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    # Dropout for regularization
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # Bottleneck representation
    rep = layers.Dense(32, activation="relu", name="student_representation")(x)

    # Final binary prediction layer with sigmoid activation
    outputs = layers.Dense(1, activation="sigmoid", name="classification_output")(rep)

    # Defining the audio-only student classification model
    model = Model(inputs=audio_input, outputs=outputs, name="audio_student_classifier")

    return model


class LUPIStudentClassifier(tf.keras.Model):
    def __init__(self, student_model, teacher_prediction_model, alpha=0.5, name="lupi_student_classifier"):
        # Wrapper model that trains a student with optional teacher prediction distillation (KL loss)
        super().__init__(name=name)
        self.student_model = student_model
        self.teacher_prediction_model = teacher_prediction_model
        # Alpha controls weighting between BCE (true labels) and KL (teacher guidance)
        self.alpha = float(alpha)

        # Trackers for reporting losses during evaluation
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.bce_tracker  = tf.keras.metrics.Mean(name="bce_loss")
        self.kl_tracker  = tf.keras.metrics.Mean(name="kl_loss")
        
        # KL divergence function
        self.kl_fn = tf.keras.losses.KLDivergence()
        
    @property
    def metrics(self):
        return [self.loss_tracker, self.bce_tracker, self.kl_tracker]

    def call(self, audio_x, training=False):
        # Forward pass uses the underlying student model
        return self.student_model(audio_x, training=training)
    
    def train_step(self, data):
        # Unpacking the data where x contains modalities and y is the target label
        x, y = data

        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)

        # Audio is used by the student while video and physio are only used to compute the frozen teacher predictions
        audio_x = x["audio"]
        video_x = x["video"]
        physio_x = x["physio"]

        with tf.GradientTape() as tape:
            # Student prediction probability
            student_probability = self.student_model(audio_x, training=True)

            # Task loss: Binary cross-entropy between true labels and student predictions
            bce = tf.keras.losses.binary_crossentropy(y, student_probability)
            bce_loss = tf.reduce_mean(bce)

            if self.alpha == 0.0:
                # If alpha is 0, teacher guidance is ignored and we train normally
                kl_loss = tf.constant(0.0, dtype=tf.float32)
                total_loss = bce_loss
            else:
                # Teacher prediction probability from all modalities (privileged information)
                teacher_probability = self.teacher_prediction_model([audio_x, video_x, physio_x], training=False)

                # Clip probabilities to avoid log(0) issues in KL divergence
                eps = 1e-7
                teacher_probability = tf.clip_by_value(teacher_probability, eps, 1.0 - eps)
                student_probability = tf.clip_by_value(student_probability, eps, 1.0 - eps)

                # Converting sigmoid probabilities into 2 class distributions: [P(class0), P(class1)]
                teacher_distribution = tf.concat([1.0 - teacher_probability, teacher_probability], axis=1)
                student_distribution = tf.concat([1.0 - student_probability, student_probability], axis=1)

                # KL divergence between teacher and student distributions
                kl_loss = self.kl_fn(teacher_distribution, student_distribution)
                
                # Weighted combination of BCE and KL based on alpha
                total_loss = (1.0 - self.alpha) * bce_loss + self.alpha * kl_loss

        # Backpropagation on the student model's trainable variables
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Updating trackers
        self.loss_tracker.update_state(total_loss)
        self.bce_tracker.update_state(bce_loss)
        self.kl_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # Evaluation step using audio-only input (no privileged information)
        x, y = data

        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)
        
        # The input can be either a dictionary or raw arrays
        if isinstance(x, dict):
            audio_x = x["audio"]
        else:
            audio_x = x

        # Student prediction probability
        student_probability = self.student_model(audio_x, training=False)

        # Standard BCE evaluation loss on held-out participant
        bce = tf.keras.losses.binary_crossentropy(y, student_probability)
        bce_loss = tf.reduce_mean(bce)

        # KL loss is not computed at test time
        kl_loss = tf.constant(0.0, dtype=tf.float32)
        total_loss = bce_loss
        
        # Updating trackers
        self.loss_tracker.update_state(total_loss)
        self.bce_tracker.update_state(bce_loss)
        self.kl_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}
    

# Loading the csv files
audio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/audio_data.csv")
video_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/video_data.csv")
physio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/physio_data.csv")

# Participant label
groups = audio_df["Participant"].astype(str).values

# Target variables
y_arousal = audio_df["arousal_class"].values.astype(int)
y_valence = audio_df["valence_class"].values.astype(int)

# Feature matrices for each modality
X_audio = audio_df.drop(
    columns=["Participant", "median_arousal", "median_valence", "arousal_class", "valence_class"]).values

X_video = video_df.drop(
    columns=["Participant", "median_arousal", "median_valence", "arousal_class", "valence_class"]).values

X_physio = physio_df.drop(
    columns=["Participant", "median_arousal", "median_valence", "arousal_class", "valence_class"]).values


def train_student_classifier(X_audio, X_video, X_physio, y, groups, target_name="arousal", alpha=0.0, student_epochs=50, student_batch_size=32, teacher_epochs=200, teacher_batch_size=64, teacher_patience=15, student_patience=8):
    # LOPO protocol using GroupKFold to ensure each fold leaves a single participant out
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)

    # Storing test results for each fold
    test_accuracy_scores = []
    test_f1_scores = []
    test_pearson_scores = []

    print(f"\nTraining student classifier for target: {target_name} with alpha: {alpha}")
    print(f"LOPO folds: {n_splits}")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_audio, y, groups=groups), start=1):
        print(f"\nFold {fold}/{n_splits}")

        fold_dir = os.path.join(RUN_DIR, f"{target_name}", f"alpha_{alpha}", f"fold_{fold:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        # Splitting audio and targets into train and test sets
        X_audio_train, X_audio_test = X_audio[train_idx], X_audio[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Early stopping for student training
        early_stopping_student = EarlyStopping(monitor="val_loss", patience=student_patience, restore_best_weights=True)

        if alpha == 0.0:
            # Baseline student training without teacher guidance
            student = build_audio_student_classification_model(X_audio_train.shape[1])
            student.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")

            student.fit(
                X_audio_train, y_train,
                validation_split=0.2,
                epochs=student_epochs,
                batch_size=student_batch_size,
                callbacks=[early_stopping_student],
                verbose=0)

            student.save(os.path.join(fold_dir, "student.keras"))

            # Probabilistic prediction on held-out participant
            y_prob = student.predict(X_audio_test, verbose=0).flatten()

        else:
            # Additional modalities available for teacher training
            X_video_train = X_video[train_idx]
            X_physio_train = X_physio[train_idx]

            # Building and training the multimodal teacher on this fold
            input_shapes = (X_audio_train.shape[1], X_video_train.shape[1], X_physio_train.shape[1])
            teacher_model = build_teacher_classification(input_shapes, target_name=target_name)

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
            
            # Extracting the teacher's probabilistic outputs
            teacher_prediction_model = tf.keras.Model(
                inputs=teacher_model.inputs,
                outputs=teacher_model.output,
                name="teacher_prediction_model")

            teacher_prediction_model.trainable = False

            # Build student and wrap with LUPI training logic
            student = build_audio_student_classification_model(X_audio_train.shape[1])
            lupi_model = LUPIStudentClassifier(student, teacher_prediction_model, alpha=alpha)
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

            # Saving the student and teacher models for this fold
            student.save(os.path.join(fold_dir, "student.keras"))
            teacher_model.save(os.path.join(fold_dir, "teacher.keras"))

            # Probabilistic prediction using audio-only path at test time
            y_prob = lupi_model.predict(X_audio_test, verbose=0).flatten()

        # Converting probabilities to class labels using a threshold of 0.5
        y_pred = (y_prob >= 0.5).astype(int)

        np.save(os.path.join(fold_dir, "y_test.npy"), y_test)
        np.save(os.path.join(fold_dir, "y_prob.npy"), y_prob)
        np.save(os.path.join(fold_dir, "y_pred.npy"), y_pred)
        np.save(os.path.join(fold_dir, "test_idx.npy"), test_idx)
        np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)

        meta = {
            "fold": fold,
            "target_name": target_name,
            "alpha": float(alpha),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "threshold": 0.5,
            "distillation": "KL",
        }
        with open(os.path.join(fold_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Evaluating: Accuracy, F1, Pearson on probabilistic outputs
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        r, _ = pearsonr(y_test, y_prob)
        if np.isnan(r):
            r = np.nan

        test_accuracy_scores.append(accuracy)
        test_f1_scores.append(f1)
        test_pearson_scores.append(r)

        print(f"Test Accuracy: {accuracy:.6f}, Test F1: {f1:.6f}, Test Pearson r: {r:.6f}")

    # Aggregating metrics across folds
    mean_accuracy = float(np.mean(test_accuracy_scores))
    std_acc = float(np.std(test_accuracy_scores))

    mean_f1 = float(np.mean(test_f1_scores))
    std_f1 = float(np.std(test_f1_scores))

    mean_pearson = float(np.nanmean(test_pearson_scores))
    std_pearson = float(np.nanstd(test_pearson_scores))
    valid_pearson = int(np.sum(~np.isnan(test_pearson_scores)))

    print(f"\nFinal Results for target: {target_name} with alpha: {alpha}")
    print(f"Test Accuracy: {mean_accuracy:.6f} ± {std_acc:.6f}")
    print(f"Test F1:       {mean_f1:.6f} ± {std_f1:.6f}")
    print(f"Test Pearson (valid folds {valid_pearson}/{n_splits}): {mean_pearson:.6f} ± {std_pearson:.6f}")

    # Saving results to a CSV file
    results_df = pd.DataFrame({
        "Fold": list(range(1, n_splits + 1)),
        "Test_Accuracy": test_accuracy_scores,
        "Test_F1": test_f1_scores,
        "Test_Pearson": test_pearson_scores})

    results_df.loc[n_splits] = ["Mean ± Std", f"{mean_accuracy:.6f} ± {std_acc:.6f}", f"{mean_f1:.6f} ± {std_f1:.6f}", f"{mean_pearson:.6f} ± {std_pearson:.6f}"]

    # Exporting results to CSV
    out_csv = f"audio_student_classifier_KL_Divergence_{target_name}_alpha_{alpha}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


# Running the training for both arousal and valence targets across different alpha values to compare the effect of LUPI on student performance
for alpha in [0.25, 0.5, 0.75, 1.0]:
    for target_name, y_target in {"arousal": y_arousal, "valence": y_valence}.items():
        train_student_classifier(
            X_audio=X_audio, X_video=X_video, X_physio=X_physio,
            y=y_target, groups=groups,
            target_name=target_name,
            alpha=alpha,
            student_epochs=50,
            student_batch_size=32)