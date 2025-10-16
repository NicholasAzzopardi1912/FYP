import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Loading the Audio Modality
audio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/audio_data.csv")

# Splitting the data into features and labels
X = audio_df.drop(columns=["median_arousal", "median_valence", "arousal_class", "valence_class"])
y = audio_df["arousal_class, valence_class"]

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

# Evaluating the model on the test data and printing the loss and accuracy
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Interpreting the training vs validation accuracy over epochs
# It basically shows how well the model is learning to classify the data in comparison to the validation set
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Audio Modality Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Interpreting the training vs validation loss over epochs
# Helps to determine overfitting during training as the models error evolves when making predictions
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Audio Modality Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy')
plt.legend()
plt.show()