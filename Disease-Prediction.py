# -*- coding: utf-8 -*-
"""
task4.py

This script trains a Random Forest Classifier to predict disease outcomes
based on symptoms and patient profiles.
"""

# **Installing Libraries**
# Make sure you have these libraries installed. 
# You can run this in your command line:
# pip install pandas scikit-learn

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# **Load the Dataset**
# Load the dataset
df = pd.read_csv(r'C:\Users\salee\Downloads\Dataset\Disease_symptom_and_patient_profile_dataset.csv')  # Make sure the dataset path is correct
print(df.head())
print(df.columns)

# **Preprocess the Data**
# Encode categorical variables
label_encoders = {}
for column in ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 
               'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Verify the DataFrame after encoding
print(df.head())
print(df.columns)

# Split the data into features and target
X = df.drop('Outcome Variable', axis=1)
y = df['Outcome Variable']

# **Split the Data**
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Train a Classification Model**
# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# **Evaluate the Model**
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# **Making Predictions with New Data**
def make_prediction(new_data):
    """Encodes new data, makes a prediction, and returns the decoded outcome."""
    new_data_encoded = {}
    for column, value in new_data.items():
        if column in label_encoders:
            new_data_encoded[column] = label_encoders[column].transform([value])[0]
        else:
            new_data_encoded[column] = value

    # Convert the encoded new data into a DataFrame with the same columns as X_train
    new_data_df = pd.DataFrame([new_data_encoded], columns=X.columns)

    # Make a prediction
    prediction = model.predict(new_data_df)
    
    # Decode the prediction
    prediction_decoded = label_encoders['Outcome Variable'].inverse_transform(prediction)
    return prediction_decoded[0]

# Example of making predictions with new data
new_data_example_1 = {
    'Disease': 'Influenza',
    'Fever': 'Yes',
    'Cough': 'No',
    'Fatigue': 'Yes',
    'Difficulty Breathing': 'Yes',
    'Age': 20,
    'Gender': 'Female',
    'Blood Pressure': 'Low',
    'Cholesterol Level': 'Normal'
}

prediction_1 = make_prediction(new_data_example_1)
print(f'Prediction for example 1: {prediction_1}')

new_data_example_2 = {
    'Disease': 'Common Cold',
    'Fever': 'No',
    'Cough': 'Yes',
    'Fatigue': 'Yes',
    'Difficulty Breathing': 'No',
    'Age': 25,
    'Gender': 'Female',
    'Blood Pressure': 'Normal',
    'Cholesterol Level': 'Normal'
}

prediction_2 = make_prediction(new_data_example_2)
print(f'Prediction for example 2: {prediction_2}')
