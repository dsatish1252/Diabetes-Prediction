import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Display columns and types for verification
print(df.columns)
print(df.dtypes)
print(df.head())

# Encode categorical columns
# Ensure these mappings match your dataset
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['smoking_history'] = df['smoking_history'].map({
    'never': 0, 'No Info': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5
})

# Handle missing values for numeric columns
df.fillna(df.mean(numeric_only=True), inplace=True)

# Handle missing or invalid categorical values (if needed)
df['smoking_history'].fillna(0, inplace=True)  # Example: fill with 'never' if needed

# Drop rows with any remaining NaNs (in case of categorical issues)
df.dropna(inplace=True)

# Check again if any non-numeric columns remain after encoding
non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns
if len(non_numeric_cols) > 0:
    print("Please encode these columns too:", non_numeric_cols)
    exit()

# Separate features and target
X = df.drop(columns=['diabetes'])
y = df['diabetes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
log_clf = LogisticRegression().fit(X_train_scaled, y_train)
rf_clf = RandomForestClassifier().fit(X_train_scaled, y_train)
gb_clf = GradientBoostingClassifier().fit(X_train_scaled, y_train)
knn_clf = KNeighborsClassifier().fit(X_train_scaled, y_train)

# Ensemble: average probabilities
log_prob = log_clf.predict_proba(X_test_scaled)
rf_prob = rf_clf.predict_proba(X_test_scaled)
gb_prob = gb_clf.predict_proba(X_test_scaled)
knn_prob = knn_clf.predict_proba(X_test_scaled)

# Average the probabilities from all models
avg_prob = (log_prob + rf_prob + gb_prob + knn_prob) / 4
final_preds = np.argmax(avg_prob, axis=1)

# Evaluation
print("Accuracy:", accuracy_score(y_test, final_preds))
print("Classification Report:\n", classification_report(y_test, final_preds))

# Save models and scaler for later use
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(log_clf, 'logistic_model.pkl')
joblib.dump(rf_clf, 'random_forest_model.pkl')
joblib.dump(gb_clf, 'gradient_boost_model.pkl')
joblib.dump(knn_clf, 'knn_model.pkl')

print("All models and scaler saved successfully.")
