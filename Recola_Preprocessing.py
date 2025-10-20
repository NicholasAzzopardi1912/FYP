# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Loading the CSV file into a pandas DataFrame
file_path = "C:/Users/nicho/OneDrive/University/Year 3/FYP/all_participants_data.csv"
df = pd.read_csv(file_path)

df['Participant'] = df['Participant'].astype('str')

target_columns = ['median_arousal', 'median_valence']
ID_column = ['Participant']

# Extracting all the required features except for 'Participant', 'median_arousal', and 'median_valence'
features = [col for col in df.columns if col not in target_columns + [ID_column]]

# Extracting the features and target variables for regression
X = df[features]

y_arousal = df['median_arousal']
y_valence = df['median_valence']

# Standardizing the features to ensure that they all have an equal contribution to the model
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features) # Array goes back to Dataframe

# Adding back the ID column to the scaled features DataFrame
X_scaled_df[ID_column] = df[ID_column].values

# Computing the median values for arousal and valence to create binary classification targets
median_arousal = y_arousal.median()
median_valence = y_valence.median()

# Binary classification labels: 1 = high, 0 = low
y_arousal_class = (y_arousal > median_arousal).astype(int)
y_valence_class = (y_valence > median_valence).astype(int)


# Printing a summary of the dataset
summary = {
    "Num features": X_scaled_df.shape[1],
    "Num samples": X_scaled_df.shape[0],
    "Arousal median": median_arousal,
    "Valence median": median_valence,
    "Arousal class distribution": y_arousal_class.value_counts().to_dict(),
    "Valence class distribution": y_valence_class.value_counts().to_dict()
}

print("\n DATA SUMMARY")
for key, value in summary.items():
    print(f"{key}: {value}")

# Combining the processed features and target variables into a single DataFrame and saving to a new CSV file
processed_df = X_scaled_df.copy()
processed_df['median_arousal'] = y_arousal
processed_df['median_valence'] = y_valence
processed_df['arousal_class'] = y_arousal_class
processed_df['valence_class'] = y_valence_class

save_path = "C:/Users/nicho/OneDrive/University/Year 3/FYP/processed_recola.csv"
processed_df.to_csv(save_path, index=False)
