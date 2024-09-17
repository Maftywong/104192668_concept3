# Necessary imports
import pandas as pd
from urllib.parse import urlparse
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
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

# Extract top-level domain (TLD) and handle invalid URLs
def extract_tld(url):
    netloc = urlparse(url).netloc.split('.')
    if len(netloc) > 1:
        return netloc[-1]
    return 'other'  # Default value for invalid or non-standard URLs

df['tld'] = df['url'].apply(extract_tld)

# Verify the 'tld' column
print(df['tld'].head())

# Limit to top TLDs to avoid excessive one-hot encoding
top_tlds = df['tld'].value_counts().index[:10]  # Adjust this value based on memory
df['tld'] = df['tld'].apply(lambda x: x if x in top_tlds else 'other')

# Now apply one-hot encoding
df = pd.get_dummies(df, columns=['tld'], drop_first=True)

# Drop the original 'url' column as it is not useful anymore
df = df.drop(columns=['url'])

# Encode the target labels (mapping 'benign', 'phishing', 'defacement', 'malware' to 0, 1, 2, 3 respectively)
label_mapping = {'benign': 0, 'phishing': 1, 'defacement': 2, 'malware': 3}
df['type'] = df['type'].map(label_mapping)

# Ensure all columns are numerical (print data types for verification)
print(df.dtypes)

# Define the features (X) and target (y)
X = df.drop(columns=['type'])  # Features
y = df['type']  # Target

# Split the dataset using stratified sampling to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale the training and testing sets
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the K-Nearest Neighbors (KNN) model
knn_model = KNeighborsClassifier(n_neighbors=5)  # Default k=5, can be tuned later
knn_model.fit(X_train, y_train)

# Evaluate the model
y_pred = knn_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save the trained KNN model
joblib.dump(knn_model, 'knn_model.pkl')
