import pandas as pd

# Load the processed dataset
df = pd.read_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/processed_recola.csv")

# Define modalities based on column name patterns
audio_features = [col for col in df.columns if col.startswith("ComParE") or col == "audio_speech_probability_lstm_vad"]
video_features = [col for col in df.columns if col.startswith("VIDEO") or col == "Face_detection_probability"]
physio_features = [col for col in df.columns if col.startswith("ECG") or col.startswith("EDA")]

# Extract target columns
target_columns = ['median_arousal', 'median_valence', 'arousal_class', 'valence_class']

# Create separate DataFrames for each modality
audio_df = df[audio_features + target_columns]
video_df = df[video_features + target_columns]
physio_df = df[physio_features + target_columns]

# Save the modality-specific DataFrames to new CSV files
audio_df.to_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/audio_data.csv", index=False)
video_df.to_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/video_data.csv", index=False)
physio_df.to_csv("C:/Users/nicho/OneDrive/University/Year 3/FYP/physio_data.csv", index=False)

print("CSV files saved successfully.")