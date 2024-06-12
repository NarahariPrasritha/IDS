# Import necessary libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

# Load your dataset
data = pd.read_csv('UNSW_NB15_no_outliers.csv')

# Assuming you already have a subset of selected features in a list called 'selected_features'
selected_features = ['rate', 'dttl', 'dload', 'swin', 'dwin', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_dst_src_ltm', 'ct_srv_dst']

X = data[selected_features].values  # Features
y = data['label'].values  # Target variable

# Feature scaling for better performance
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Initialize the SVM classifier
base_classifier = SVC(kernel='linear', random_state=42)

# Initialize the Bagging classifier with SVM as the base estimator
ensemble_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the ensemble classifier
ensemble_classifier.fit(X_scaled, y)

# Initialize Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    features = []
    for feature in selected_features:
        features.append(float(request.form[feature]))

    # Preprocess the input features
    input_features = sc.transform([features])

    # Make prediction
    prediction = ensemble_classifier.predict(input_features)

    # Return the prediction result
    if prediction == 0:
        return render_template('index.html', prediction_text='Normal Activity')
    else:
        return render_template('index.html', prediction_text='Intrusion Detected')

if __name__ == "__main__":
    app.run(debug=True)

