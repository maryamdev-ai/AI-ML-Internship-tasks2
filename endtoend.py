import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

# Load the dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Display basic info
print(df.info())
print(df.head())

# Convert TotalCharges to numeric (it's loaded as object due to empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values (the few rows with empty TotalCharges are new customers with 0 tenure)
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Convert target variable to binary
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop customer ID as it's not a useful feature
df = df.drop('customerID', axis=1)

# Split into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify numeric and categorical columns
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

# Create transformers for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a function to build different model pipelines
def create_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# Define models to try
models = {
    'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
    'random_forest': RandomForestClassifier(random_state=42)
}

# Create pipelines for each model
pipelines = {name: create_pipeline(model) for name, model in models.items()}

# Define parameter grids for each model
param_grids = {
    'logistic_regression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'liblinear']
    },
    'random_forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
}

# Perform GridSearchCV for each model
best_models = {}
for name in pipelines.keys():
    print(f"\nTraining {name}...")
    grid_search = GridSearchCV(
        pipelines[name],
        param_grids[name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = best_models[name].predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Let's assume Random Forest performed better (in practice, check your results)
best_pipeline = best_models['random_forest']

# Save the complete pipeline to disk
dump(best_pipeline, 'churn_pipeline.joblib')

print("Pipeline saved successfully!")

from joblib import load

# Load the pipeline
pipeline = load('churn_pipeline.joblib')

# Prepare new data (in the same format as training data)
new_data = pd.DataFrame({
    'tenure': [12],
    'MonthlyCharges': [70.50],
    'TotalCharges': [850.00],
    'gender': ['Female'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'PhoneService': ['Yes'],
    # ... include all other features
})

# Make prediction
prediction = pipeline.predict(new_data)
prediction_proba = pipeline.predict_proba(new_data)

print(f"Churn prediction: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability: {prediction_proba[0][1]:.2f}")