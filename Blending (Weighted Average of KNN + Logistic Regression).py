# Import necessary libraries
import pandas as pd
from urllib.parse import urlparse
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Load the dataset
df = pd.read_csv('malicious_phish.csv')

# Feature extraction from 'url'
df['url_length'] = df['url'].apply(len)
df['domain_length'] = df['url'].apply(lambda x: len(urlparse(x).netloc))
df['dot_count'] = df['url'].apply(lambda x: x.count('.'))
df['at_count'] = df['url'].apply(lambda x: x.count('@'))
df['slash_count'] = df['url'].apply(lambda x: x.count('/'))
df['question_mark_count'] = df['url'].apply(lambda x: x.count('?'))
df['suspicious_keywords'] = df['url'].apply(lambda x: 1 if any(kw in x.lower() for kw in ['login', 'secure', 'account', 'update', 'bank', 'signin']) else 0)
df['has_ip_address'] = df['url'].apply(lambda x: 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', x) else 0)
df['subdomain_count'] = df['url'].apply(lambda x: len(urlparse(x).netloc.split('.')) - 2)

# Function to extract TLD (Top Level Domain)
def extract_tld(url):
    netloc = urlparse(url).netloc.split('.')
    if len(netloc) > 1:
        return netloc[-1]  # Return the TLD
    return 'other'

df['tld'] = df['url'].apply(extract_tld)

# Limit the number of TLDs to the most common ones
top_tlds = df['tld'].value_counts().index[:10]
df['tld'] = df['tld'].apply(lambda x: x if x in top_tlds else 'other')

# Apply one-hot encoding to the 'tld' column
df = pd.get_dummies(df, columns=['tld'], drop_first=True)

# Drop the original 'url' column as it's no longer needed
df = df.drop(columns=['url'])

# Encode the target labels (benign, phishing, defacement, malware)
label_mapping = {'benign': 0, 'phishing': 1, 'defacement': 2, 'malware': 3}
df['type'] = df['type'].map(label_mapping)

# Define the features (X) and target (y)
X = df.drop(columns=['type'])  # Features
y = df['type']  # Target

# Split the dataset into training and testing sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features for KNN and Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Train the Logistic Regression model with increased max_iter and different solver
lr_model = LogisticRegression(max_iter=5000, solver='liblinear', random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Get the probability predictions for both models
knn_probs = knn_model.predict_proba(X_test_scaled)
lr_probs = lr_model.predict_proba(X_test_scaled)

# Blend the predictions by taking the weighted average of both models' predictions
blended_probs = (0.5 * knn_probs) + (0.5 * lr_probs)

# Make final predictions by choosing the class with the highest probability
y_pred = np.argmax(blended_probs, axis=1)

# Evaluate the blended model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save the blended model
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(lr_model, 'lr_model.pkl')

from google.colab import files

# Download the KNN model
files.download('knn_model.pkl')

# Download the Logistic Regression model
files.download('lr_model.pkl')
