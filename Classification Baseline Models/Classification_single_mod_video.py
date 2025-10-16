import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Loading the Video Modality
audio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/video_data.csv")

# Splitting the data into features and labels
X = audio_df.drop(columns=["median_arousal", "median_valence", "arousal_class", "valence_class"])
y = audio_df[["arousal_class", "valence_class"]]

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Developing the Dense Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),  # Input layer
    Dropout(0.3),   # Dropout layer to prevent overfitting at 30%
    Dense(64, activation='relu'), # Hidden layer
    Dropout(0.3), # Dropout layer to prevent overfitting at 30%
    Dense(32, activation='relu'), # Hidden layer
    Dense(2, activation='sigmoid') # Output layer for binary classification
])

# Compiling the model using the Adam optimizer and binary cross-entropy loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the model to the training data, setting the epochs to 50 and batch size to 32, along with a validation split of 20%
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose = 1)

# On average, across both arousal and valence outputs and across all test samples, the binary cross-entropy error is produced
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nOverall Test Loss: {loss:.4f}")
print(f"Overall Test Accuracy (combined): {accuracy:.4f}")

# Generating predictions and converting probabilities to binary class labels
y_pred = model.predict(x_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Computing per target metrics
arousal_true = y_test.iloc[:, 0]
valence_true = y_test.iloc[:, 1]
arousal_pred = y_pred_binary[:, 0]
valence_pred = y_pred_binary[:, 1]

# Displaying classification reports for both arousal and valence
print("\n Arousal Classification Metrics ")
print("Accuracy:", accuracy_score(arousal_true, arousal_pred))
print(classification_report(arousal_true, arousal_pred, digits=3))

print("\n Valence Classification Metrics ")
print("Accuracy:", accuracy_score(valence_true, valence_pred))
print(classification_report(valence_true, valence_pred, digits=3))

# Interpreting the training vs validation accuracy over epochs
# It basically shows how well the model is learning to classify the data in comparison to the validation set
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Video Modality Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Interpreting the training vs validation loss over epochs
# Helps to determine overfitting during training as the models error evolves when making predictions
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Video Modality Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy')
plt.legend()
plt.show()