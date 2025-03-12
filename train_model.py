import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
career_data = pd.read_csv("career_dataset.csv")  # Make sure this file exists

# Features and labels
X = career_data.iloc[:, :-1].values  
y = career_data.iloc[:, -1].values  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save model
joblib.dump(knn, "career_prediction_model.pkl")
print("Model trained and saved as career_prediction_model.pkl")
