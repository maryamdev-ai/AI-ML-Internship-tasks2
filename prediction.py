"""
House Price Prediction Script - Customized for Your Dataset
Author: Maryam Abdullah
Date: 6/24/2025
Description: Predicts house prices using property features from your specific dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import joblib

# Set matplotlib to display in VS Code
plt.switch_backend('module://matplotlib_inline.backend_inline')

def load_data(file_path):
    """Load the dataset from CSV file."""
    try:
        df = pd.read_csv("train.csv")
        print("Data loaded successfully!")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data by selecting features and handling missing values."""
    print("\nAvailable columns in your dataset:")
    print(df.columns.tolist())
    
    # Define features based on your actual dataset columns
    features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Location', 'Condition', 'Garage']
    target = 'Price'
    
    # Verify which features actually exist in the dataset
    available_features = [col for col in features if col in df.columns]
    missing_features = [col for col in features if col not in df.columns]
    
    if missing_features:
        print(f"\nWarning: The following features are missing: {missing_features}")
    
    print(f"\nUsing features: {available_features}")
    print(f"Using target: {target}")
    
    # Create new dataframe with selected features
    data = df[available_features + [target]].copy()
    
    # Define numerical and categorical features
    numerical_features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']
    categorical_features = ['Location', 'Condition', 'Garage']
    
    # Only keep features that actually exist in our data
    numerical_features = [col for col in numerical_features if col in available_features]
    categorical_features = [col for col in categorical_features if col in available_features]
    
    print(f"\nNumerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Create transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    # Only create categorical transformer if we have categorical features
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)])
    else:
        # If no categorical features, just use numerical transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features)])
    
    return data, preprocessor, numerical_features, categorical_features, target

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate models."""
    # Linear Regression
    print("\nTraining Linear Regression model...")
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())])
    
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    
    # Gradient Boosting Regression
    print("Training Gradient Boosting model...")
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))])
    
    gb_pipeline.fit(X_train, y_train)
    y_pred_gb = gb_pipeline.predict(X_test)
    
    return lr_pipeline, gb_pipeline, y_pred_lr, y_pred_gb

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance and print metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\n{model_name} Evaluation:")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    
    return mae, rmse

def plot_results(y_test, y_pred_lr, y_pred_gb):
    """Create visualization plots."""
    plt.figure(figsize=(15, 6))
    
    # Scatter plot of actual vs predicted values
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred_lr)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Linear Regression')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_test, y=y_pred_gb)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Gradient Boosting')
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    print("\nSaved prediction comparison plot as 'prediction_comparison.png'")
    plt.show()

def plot_feature_importance(pipeline, numerical_features, categorical_features):
    """Plot feature importance for Gradient Boosting model."""
    try:
        # Get feature names
        feature_names = numerical_features.copy()
        
        # Add categorical feature names if they exist
        if categorical_features:
            # Get the OneHotEncoder from the pipeline
            ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            categorical_feature_names = ohe.get_feature_names_out(categorical_features)
            feature_names.extend(categorical_feature_names)
        
        # Get feature importances
        importances = pipeline.named_steps['regressor'].feature_importances_
        
        # Create a DataFrame for visualization
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title('Top Feature Importances')
        plt.savefig('feature_importance.png')
        print("Saved feature importance plot as 'feature_importance.png'")
        plt.show()
    except Exception as e:
        print(f"\nCould not create feature importance plot: {e}")

def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"\nModel saved as {filename}")

def main():
    print("House Price Prediction Script - Customized for Your Dataset")
    print("=========================================================\n")
    
    # Load data
    data_path = 'train.csv'  # Update this path if needed
    df = load_data(data_path)
    
    if df is None:
        return
    
    # Preprocess data
    data, preprocessor, numerical_features, categorical_features, target = preprocess_data(df)
    
    # Split into features and target
    X = data.drop(target, axis=1)
    y = data[target]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    lr_pipeline, gb_pipeline, y_pred_lr, y_pred_gb = train_and_evaluate(
        X_train, X_test, y_train, y_test, preprocessor)
    
    # Evaluate models
    lr_mae, lr_rmse = evaluate_model(y_test, y_pred_lr, "Linear Regression")
    gb_mae, gb_rmse = evaluate_model(y_test, y_pred_gb, "Gradient Boosting")
    
    # Visualize results
    print("\nGenerating visualizations...")
    plot_results(y_test, y_pred_lr, y_pred_gb)
    
    # Plot feature importance (only if we have features)
    if numerical_features or categorical_features:
        plot_feature_importance(gb_pipeline, numerical_features, categorical_features)
    
    # Save the best model
    save_model(gb_pipeline, 'house_price_predictor.pkl')
    
    print("\nScript execution completed successfully!")

if __name__ == "__main__":
    main()