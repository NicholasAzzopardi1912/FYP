# Classification Student Model (Representation Alignment via Cosine Distance)

# Importing Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

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
        # Dropout layer for regularization
        x = layers.Dropout(dropout, name=f"{name_prefix}_dropout_{i+1}")(x)
    # Returning the input and encoded representation of the branch
    return inputs, x

 
def build_teacher_classification(input_shapes, target_name="arousal", branch_hidden=(128, 64), fusion_hidden=(128, 32), dropout=0.3, l2=1e-4):
    # Unpacking input shapes for each modality
    audio_dimensions, video_dimensions, physiological_dimensions = input_shapes

    # Creating separate encoder branches for each modality
    audio_input, audio_encoded = encoder_branch("audio", audio_dimensions, branch_hidden, dropout, l2)
    video_input, video_encoded = encoder_branch("video", video_dimensions, branch_hidden, dropout, l2)
    physio_input, physio_encoded = encoder_branch("physio", physiological_dimensions, branch_hidden, dropout, l2)

    # Fusion Layer: Concatenating the encoded outputs from all branches
    fused = layers.Concatenate(name="fusion_layer")([audio_encoded, video_encoded, physio_encoded])

    # Fusion head to apply further processing after fusion 
    x = fused
    for i, units in enumerate(fusion_hidden):
        x = layers.Dense(units, activation="relu",
                        kernel_regularizer=regularizers.l2(l2),
                        name=f"fusion_dense_{i+1}")(x)
        
        x = layers.Dropout(dropout, name=f"fusion_dropout_{i+1}")(x)

    # Output layer for binary classification with sigmoid activation
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

    # Bottleneck representation used for LUPI alignment with teacher
    rep = layers.Dense(32, activation="relu", name="student_representation")(x)

    # Final binary prediction layer with sigmoid activation
    outputs = layers.Dense(1, activation="sigmoid", name="classification_output")(rep)

    # Defining the audio-only student classification model
    model = Model(inputs=audio_input, outputs=outputs, name="audio_student_classifier")

    return model

def cosine_distance(T, S):
    # L2 normalise both representations to compute cosine similarity
    T_normalised = tf.nn.l2_normalize(T, axis=1)
    S_normalised = tf.nn.l2_normalize(S, axis=1)

    # Cosine similarity is the dot product of the normalised vectors
    cosine_similarity = tf.reduce_sum(T_normalised * S_normalised, axis=1)

    # Converting similarity to distance
    cosine_distance = 1.0 - cosine_similarity

    # Mean distance across the batch
    return tf.reduce_mean(cosine_distance)


class LUPIStudentClassifier(tf.keras.Model):
    def __init__(self, student_model, teacher_representation_model, alpha=0.5, name="lupi_student_classifier"):
        # Wrapper model that trains a student with optional teacher representation loss
        super().__init__(name=name)
        self.student_model = student_model
        self.teacher_representation_model = teacher_representation_model
        # Alpha controls the bias between task loss and representation loss
        self.alpha = float(alpha)

        # Sub model to extract the students bottleneck representation
        self.student_rep_model = tf.keras.Model(
            inputs=self.student_model.input,
            outputs=self.student_model.get_layer("student_representation").output)

        # Trackers for reporting losses during evaluation
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.bce_tracker  = tf.keras.metrics.Mean(name="bce_loss")
        self.rep_tracker  = tf.keras.metrics.Mean(name="representation_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.bce_tracker, self.rep_tracker]

    def call(self, audio_x, training=False):
        # Forward pass uses the underlying student model
        return self.student_model(audio_x, training=training)
    
    def train_step(self, data):
        # Unpacking the data where x contains all modalities and y is the target label
        x, y = data

        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)

        # Audio is used by the student, while video and physio are only used by the teacher for representation learning
        audio_x = x["audio"]
        video_x = x["video"]
        physio_x = x["physio"]

        with tf.GradientTape() as tape:
            # Student prediction using audio input
            y_hat = self.student_model(audio_x, training=True)

            # Task loss being computed as binary cross-entropy between true labels and student predictions
            bce = tf.keras.losses.binary_crossentropy(y, y_hat)
            bce_loss = tf.reduce_mean(bce)

            if self.alpha == 0.0:
                # If alpha is 0, the teacher guidance is ignored and train normally
                rep_loss = tf.constant(0.0, dtype=tf.float32)
                total_loss = bce_loss
            else:
                # Teacher representation from all modalities (privileged information)
                T = self.teacher_representation_model([audio_x, video_x, physio_x], training=False)
                # Student representation from audio input
                S = self.student_rep_model(audio_x, training=True)
                
                # Representation loss is the cosine distance between the teacher and student representations
                rep_loss = cosine_distance(T, S)

                # Squaring to penalise larger representation misalignments more heavily
                rep_loss = tf.square(rep_loss)

                # Weighted combination of BCE and representation loss based on alpha
                total_loss = (1.0 - self.alpha) * bce_loss + self.alpha * rep_loss

        # Backpropagation on the student model's trainable variables
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))

        # Updating trackers
        self.loss_tracker.update_state(total_loss)
        self.bce_tracker.update_state(bce_loss)
        self.rep_tracker.update_state(rep_loss)

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

        # Student prediction
        y_hat = self.student_model(audio_x, training=False)

        # Classification loss on held out participant
        bce = tf.keras.losses.binary_crossentropy(y, y_hat)
        bce_loss = tf.reduce_mean(bce)

        # Representation loss not computed at test time
        rep_loss = tf.constant(0.0, dtype=tf.float32)
        total_loss = bce_loss
        
        # Updating trackers
        self.loss_tracker.update_state(total_loss)
        self.bce_tracker.update_state(bce_loss)
        self.rep_tracker.update_state(rep_loss)

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
            
            # Extracting the teacher fusion representation to act as privileged information for the student
            teacher_representation_model = tf.keras.Model(
                inputs=teacher_model.inputs,
                outputs=teacher_model.get_layer("fusion_dense_2").output,
                name="teacher_representation_model")
            
            teacher_representation_model.trainable = False

            # Build student and wrap with LUPI training logic
            student = build_audio_student_classification_model(X_audio_train.shape[1])
            lupi_model = LUPIStudentClassifier(student, teacher_representation_model, alpha=alpha)
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

            # Probabilistic prediction using audio-only path at test time
            y_prob = lupi_model.predict(X_audio_test, verbose=0).flatten()

        # Converting probabilities to class labels using a threshold of 0.5
        y_pred = (y_prob >= 0.5).astype(int)

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
    out_csv = f"audio_student_classifier_{target_name}_alpha_{alpha}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # Plotting fold-wise metrics to visualise performance stability across participants
    #plt.figure(figsize=(6, 4))
    #plt.plot(range(1, n_splits + 1), test_accuracy_scores, marker="o", label="Test Accuracy")
    #plt.plot(range(1, n_splits + 1), test_f1_scores, marker="o", label="Test F1")
    #plt.plot(range(1, n_splits + 1), test_pearson_scores, marker="o", label="Test Pearson r")
    #plt.title(f"Student ({target_name}) - RepAlign - alpha={alpha}")
    #plt.xlabel("Fold")
    #plt.ylabel("Score")
    #plt.grid(True, linestyle="--", alpha=0.6)
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

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