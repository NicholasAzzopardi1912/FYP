# Classification Student Model (Representation Alignment via Cosine Distance + Prediction Distillation with KL Divergence)

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

RUN_DIR = "runs_video_student_classification_combined"
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

def build_video_student_classification_model(input_dimensions):
    # Input layer for video features
    video_input = layers.Input(shape=(input_dimensions,), name="video_student_input")

    # First hidden layer for video classification
    x = layers.Dense(128, activation="relu", name="dense_128")(video_input)
    # Dropout for regularization
    x = layers.Dropout(0.3, name="dropout_1")(x)

    # Second hidden layer for video classification
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    # Dropout for regularization
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # Bottleneck representation
    rep = layers.Dense(32, activation="relu", name="student_representation")(x)

    # Final binary prediction layer with sigmoid activation
    outputs = layers.Dense(1, activation="sigmoid", name="classification_output")(rep)

    # Defining the video-only student classification model
    model = Model(inputs=video_input, outputs=outputs, name="video_student_classifier")

    return model

def cosine_distance(T, S):
    # L2 normalise both representations to compute cosine similarity
    T_normalised = tf.nn.l2_normalize(T, axis=1)
    S_normalised = tf.nn.l2_normalize(S, axis=1)

    # Cosine similarity is the dot product of the normalised vectors
    cosine_similarity = tf.reduce_sum(T_normalised * S_normalised, axis=1)

    # Convert similarity to distance
    cosine_distance = 1.0 - cosine_similarity

    # Mean distance across the batch
    return tf.reduce_mean(cosine_distance)


class LUPIStudentClassifierCombined(tf.keras.Model):
    def __init__(self, student_model, teacher_representation_model, teacher_prediction_model, alpha=0.5, name="lupi_student_classifier_combined"):
        # Wrapper model combines: BCE (labels) + representation alignment + KL distillation
        super().__init__(name=name)
        self.student_model = student_model
        self.teacher_representation_model = teacher_representation_model
        self.teacher_prediction_model = teacher_prediction_model
        # Alpha controls weight on privileged guidance vs ground truth supervision
        self.alpha = float(alpha)

        # Sub model to extract the student's bottleneck representation
        self.student_rep_model = tf.keras.Model(
            inputs=self.student_model.input,
            outputs=self.student_model.get_layer("student_representation").output)

        # Trackers for reporting losses during evaluation
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.bce_tracker  = tf.keras.metrics.Mean(name="bce_loss")
        self.rep_tracker  = tf.keras.metrics.Mean(name="representation_loss")
        self.kl_tracker   = tf.keras.metrics.Mean(name="kl_loss")

        # KL divergence function
        self.kl_fn = tf.keras.losses.KLDivergence()

    @property
    def metrics(self):
        return [self.loss_tracker, self.bce_tracker, self.rep_tracker, self.kl_tracker]

    def call(self, video_x, training=False):
        # Forward pass uses the underlying student model
        return self.student_model(video_x, training=training)
    
    def train_step(self, data):
        # Unpacking the data where x contains all modalities and y is the target label
        x, y = data

        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)

        # Video is used by the student, while audio and physio are only used by the teacher
        audio_x = x["audio"]
        video_x = x["video"]
        physio_x = x["physio"]

        with tf.GradientTape() as tape:
            # Student prediction using video input
            student_probability = self.student_model(video_x, training=True)

            # Task loss being computed as binary cross-entropy between true labels and student predictions
            bce = tf.keras.losses.binary_crossentropy(y, student_probability)
            bce_loss = tf.reduce_mean(bce)

            if self.alpha == 0.0:
                # If alpha is 0, the teacher guidance is ignored and train normally
                rep_loss = tf.constant(0.0, dtype=tf.float32)
                kl_loss = tf.constant(0.0, dtype=tf.float32)
                total_loss = bce_loss
            else:
                # Teacher representation from all modalities (privileged information)
                T = self.teacher_representation_model([audio_x, video_x, physio_x], training=False)
                # Student representation from video input
                S = self.student_rep_model(video_x, training=True)
                # Representation loss is the cosine distance between the teacher and student representations
                rep_loss = cosine_distance(T, S)
                # Squaring to penalise larger representation misalignments more heavily
                rep_loss = tf.square(rep_loss)

                # Teacher predicted probability (privileged prediction)
                teacher_probability = self.teacher_prediction_model([audio_x, video_x, physio_x], training=False)
                
                # Clip probabilities to avoid numerical issues inside KL
                eps = 1e-7
                teacher_probability = tf.clip_by_value(teacher_probability, eps, 1.0 - eps)
                student_probability = tf.clip_by_value(student_probability, eps, 1.0 - eps)

                # Convert sigmoid probability into 2-class distributions
                teacher_distribution = tf.concat([1.0 - teacher_probability, teacher_probability], axis=1)
                student_distribution = tf.concat([1.0 - student_probability, student_probability], axis=1)

                # KL divergence loss
                kl_loss = self.kl_fn(teacher_distribution, student_distribution)

                # Total loss is a weighted combination of BCE, representation alignment, and KL divergence
                total_loss = (1.0 - self.alpha) * bce_loss + self.alpha * (rep_loss + kl_loss)

        # Backpropagation on the student model's trainable variables
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Updating trackers
        self.loss_tracker.update_state(total_loss)
        self.bce_tracker.update_state(bce_loss)
        self.rep_tracker.update_state(rep_loss)
        self.kl_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # Evaluation step using video-only input (no privileged information)
        x, y = data

        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)
        
        # The input can be either a dictionary or raw arrays
        if isinstance(x, dict):
            video_x = x["video"]
        else:
            video_x = x

        # Student prediction probability
        student_probability = self.student_model(video_x, training=False)

        # Classification loss on held out participant
        bce = tf.keras.losses.binary_crossentropy(y, student_probability)
        bce_loss = tf.reduce_mean(bce)

        # Privileged losses are not computed at test time
        rep_loss = tf.constant(0.0, dtype=tf.float32)
        kl_loss = tf.constant(0.0, dtype=tf.float32)
        total_loss = bce_loss
        
        # Updating trackers
        self.loss_tracker.update_state(total_loss)
        self.bce_tracker.update_state(bce_loss)
        self.rep_tracker.update_state(rep_loss)
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


def train_student_classifier_combined(X_audio, X_video, X_physio, y, groups, target_name="arousal", alpha=0.0, student_epochs=50, student_batch_size=32, teacher_epochs=200, teacher_batch_size=64, teacher_patience=15, student_patience=8):
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

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_video, y, groups=groups), start=1):
        print(f"\nFold {fold}/{n_splits}")

        fold_dir = os.path.join(RUN_DIR, f"{target_name}", f"alpha_{alpha}", f"fold_{fold:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        # Splitting video and targets into train and test sets
        X_video_train, X_video_test = X_video[train_idx], X_video[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Early stopping for student training
        early_stopping_student = EarlyStopping(monitor="val_loss", patience=student_patience, restore_best_weights=True)

        if alpha == 0.0:
            # Baseline student training without teacher guidance
            student = build_video_student_classification_model(X_video_train.shape[1])
            student.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")

            student.fit(
                X_video_train, y_train,
                validation_split=0.2,
                epochs=student_epochs,
                batch_size=student_batch_size,
                callbacks=[early_stopping_student],
                verbose=0)

            student.save(os.path.join(fold_dir, "student.keras"))

            # Probabilistic prediction on held-out participant
            y_prob = student.predict(X_video_test, verbose=0).flatten()

        else:
            # Additional modalities available for teacher training
            X_audio_train = X_audio[train_idx]
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
            
            # Extracting the teacher fusion representation (privileged fusion features)
            teacher_representation_model = tf.keras.Model(
                inputs=teacher_model.inputs,
                outputs=teacher_model.get_layer("fusion_dense_2").output,
                name="teacher_representation_model")
            
            teacher_representation_model.trainable = False

            # Extracting the teacher's probabilistic outputs (privileged probabilities)
            teacher_prediction_model = tf.keras.Model(
                inputs=teacher_model.inputs,
                outputs=teacher_model.output,
                name="teacher_prediction_model")
            
            teacher_prediction_model.trainable = False

            # Build student and wrap with LUPI training logic
            student = build_video_student_classification_model(X_video_train.shape[1])
            lupi_model = LUPIStudentClassifierCombined(student, teacher_representation_model, teacher_prediction_model, alpha=alpha)
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

            student.save(os.path.join(fold_dir, "student.keras"))
            teacher_model.save(os.path.join(fold_dir, "teacher.keras"))
            teacher_representation_model.save(os.path.join(fold_dir, "teacher_rep.keras"))
            teacher_prediction_model.save(os.path.join(fold_dir, "teacher_pred.keras"))

            # Probabilistic prediction using video-only path at test time
            y_prob = lupi_model.predict(X_video_test, verbose=0).flatten()

        # Converting probabilities to class labels using a threshold of 0.5
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Save fold outputs
        np.save(os.path.join(fold_dir, "y_test.npy"), y_test)
        np.save(os.path.join(fold_dir, "y_prob.npy"), y_prob)
        np.save(os.path.join(fold_dir, "y_pred.npy"), y_pred)
        np.save(os.path.join(fold_dir, "test_idx.npy"), test_idx)
        np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)

        # Save metadata
        meta = {
            "fold": fold,
            "target_name": target_name,
            "alpha": float(alpha),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "threshold": 0.5,
            "setup": "combined",
            "representation_loss": "cosine_distance_squared",
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
    out_csv = f"video_student_classifier_Combined_{target_name}_alpha_{alpha}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


# Running the training for both arousal and valence targets across different alpha values to compare the effect of LUPI on student performance
for alpha in [0.25, 0.5, 0.75, 1.0]:
    for target_name, y_target in {"arousal": y_arousal, "valence": y_valence}.items():
        train_student_classifier_combined(
            X_audio=X_audio, X_video=X_video, X_physio=X_physio,
            y=y_target, groups=groups,
            target_name=target_name,
            alpha=alpha,
            student_epochs=50,
            student_batch_size=32)