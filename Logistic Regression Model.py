# Import necessary libraries
import pandas as pd
from urllib.parse import urlparse
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('malicious_phish.csv')

# Feature extraction from 'url'
# 'url_length' measures the length of the URL string
df['url_length'] = df['url'].apply(len)

# 'domain_length' measures the length of the domain part of the URL
df['domain_length'] = df['url'].apply(lambda x: len(urlparse(x).netloc))

# 'dot_count' counts the number of dots in the URL (helps detect subdomains)
df['dot_count'] = df['url'].apply(lambda x: x.count('.'))

# 'at_count' counts the number of '@' characters (unusual for a legitimate URL)
df['at_count'] = df['url'].apply(lambda x: x.count('@'))

# 'slash_count' counts the number of '/' characters in the URL
df['slash_count'] = df['url'].apply(lambda x: x.count('/'))

# 'question_mark_count' counts the number of '?' characters in the URL
df['question_mark_count'] = df['url'].apply(lambda x: x.count('?'))

# 'suspicious_keywords' looks for keywords like 'login', 'secure', etc., that are commonly used in phishing URLs
df['suspicious_keywords'] = df['url'].apply(lambda x: 1 if any(kw in x.lower() for kw in ['login', 'secure', 'account', 'update', 'bank', 'signin']) else 0)

# 'has_ip_address' checks if the URL contains an IP address (often used in malicious URLs)
df['has_ip_address'] = df['url'].apply(lambda x: 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', x) else 0)

# 'subdomain_count' calculates the number of subdomains by counting the dots in the netloc part of the URL
df['subdomain_count'] = df['url'].apply(lambda x: len(urlparse(x).netloc.split('.')) - 2)

# Function to extract the top-level domain (TLD) from the URL
def extract_tld(url):
    netloc = urlparse(url).netloc.split('.')
    if len(netloc) > 1:
        return netloc[-1]  # Get the last part of the domain (e.g., 'com', 'org')
    return 'other'  # Default to 'other' if the URL is non-standard or invalid

# Apply the TLD extraction function to the 'url' column
df['tld'] = df['url'].apply(extract_tld)

# Display the first few rows of the 'tld' column for verification
print(df['tld'].head())

# Limit the number of TLDs by grouping less common ones into the 'other' category
# This helps to avoid excessive one-hot encoding which can consume memory
top_tlds = df['tld'].value_counts().index[:10]  # Take the top 10 most frequent TLDs
df['tld'] = df['tld'].apply(lambda x: x if x in top_tlds else 'other')

# Apply one-hot encoding to the 'tld' column (each TLD becomes a separate feature/column)
# 'drop_first=True' ensures that one of the categories is dropped to avoid multicollinearity
df = pd.get_dummies(df, columns=['tld'], drop_first=True)

# Drop the original 'url' column, as it is no longer needed
df = df.drop(columns=['url'])

# Encode the target labels (converting 'benign', 'phishing', 'defacement', 'malware' to numerical values)
label_mapping = {'benign': 0, 'phishing': 1, 'defacement': 2, 'malware': 3}
df['type'] = df['type'].map(label_mapping)

# Verify that all columns are numerical by displaying the data types
print(df.dtypes)

# Define the features (X) by dropping the target column ('type')
X = df.drop(columns=['type'])

# Define the target (y), which is the 'type' column
y = df['type']

# Split the dataset into training and testing sets using stratified sampling
# Stratified sampling ensures that the class distribution is maintained in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the StandardScaler to scale the features
# Scaling ensures that all features have the same scale, which helps the logistic regression model perform better
scaler = StandardScaler()

# Apply scaling to the training data (fitting the scaler)
X_train = scaler.fit_transform(X_train)

# Apply the same scaling transformation to the test data (without fitting the scaler again)
X_test = scaler.transform(X_test)

# Initialize the Logistic Regression model
# 'max_iter=1000' ensures the model converges by allowing up to 1000 iterations during optimization
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the Logistic Regression model using the training data
log_reg_model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = log_reg_model.predict(X_test)

# Display the classification report, which includes precision, recall, and F1-score for each class
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Display the overall accuracy of the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save the trained Logistic Regression model to a file for future use
joblib.dump(log_reg_model, 'log_reg_model.pkl')

from google.colab import files

# Download the Logistic Regression model
files.download('log_reg_model.pkl')
