import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create sample dataset
data = {
    'Attendance': np.random.randint(50, 100, 300),
    'Internal_Marks': np.random.randint(30, 100, 300),
    'Study_Hours': np.random.randint(1, 6, 300),
    'Assignments': np.random.randint(1, 10, 300),
}

df = pd.DataFrame(data)

# Create result column
df['Result'] = np.where(
    (df['Attendance'] > 75) & (df['Internal_Marks'] > 50),
    1,
    0
)

# Features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)