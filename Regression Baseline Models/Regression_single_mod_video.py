import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Loading the preprocessed video dataset
audio_df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/video_data.csv")

# Splitting the dataset into features and target variables
X = audio_df.drop(columns=["median_arousal", "median_valence", "arousal_class", "valence_class"])
y = audio_df[["median_arousal", "median_valence"]]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)), # Input layer
    Dropout(0.3), # Dropout layer for regularization
    Dense(64, activation='relu'), # Hidden layer
    Dropout(0.3), # Dropout layer for regularization
    Dense(32, activation='relu'), # Hidden layer
    Dense(2, activation='linear')  # Output layer for regression (2 outputs: arousal and valence)
])

# Compiling the model using Adam optimizer and Mean Squared Error loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Training the model using epochs of size 50, batch size of 32, and a validation split of 20%
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluating the model on the test set
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"\nMean Squared Error on Test Set: {mse}")


# Plots to determine whether the model is learning and generalizing well

# Plotting training & validation loss values to determine whether the model is overfitting
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Video Modality - Training and Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Plotting training & validation to determine the average size of the error
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Video Modality - Training and Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()