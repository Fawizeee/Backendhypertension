import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# Create synthetic training data that matches the expected features
def create_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'Age': np.random.randint(18, 80, n_samples),
        'Salt_Intake': np.random.uniform(2, 15, n_samples),
        'Stress_Score': np.random.randint(1, 11, n_samples),
        'BP_History': np.random.choice(['Normal', 'High', 'Low'], n_samples, p=[0.6, 0.3, 0.1]),
        'Sleep_Duration': np.random.uniform(4, 10, n_samples),
        'BMI': np.random.uniform(18, 40, n_samples),
        'Medication': np.random.choice(['None', 'Yes'], n_samples, p=[0.7, 0.3]),
        'Family_History': np.random.choice(['No', 'Yes'], n_samples, p=[0.6, 0.4]),
        'Exercise_Level': np.random.choice(['Low', 'Moderate', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
        'Smoking_Status': np.random.choice(['Non-Smoker', 'Smoker'], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on realistic risk factors
    risk_score = (
        (df['Age'] - 18) / 62 * 0.3 +  # Age factor
        (df['Salt_Intake'] - 2) / 13 * 0.2 +  # Salt intake
        (df['Stress_Score'] - 1) / 9 * 0.2 +  # Stress
        (df['BMI'] - 18) / 22 * 0.2 +  # BMI
        (df['Sleep_Duration'] - 4) / 6 * -0.1 +  # Sleep (inverse)
        (df['BP_History'] == 'High').astype(int) * 0.3 +  # BP history
        (df['Family_History'] == 'Yes').astype(int) * 0.2 +  # Family history
        (df['Smoking_Status'] == 'Smoker').astype(int) * 0.2 +  # Smoking
        (df['Medication'] == 'Yes').astype(int) * 0.1 +  # Medication
        (df['Exercise_Level'] == 'Low').astype(int) * 0.1  # Exercise
    )
    
    # Add some noise
    risk_score += np.random.normal(0, 0.1, n_samples)
    
    # Convert to binary classification
    df['Hypertension'] = (risk_score > 0.5).astype(int)
    
    return df

def train_model():
    print("Creating synthetic training data...")
    df = create_synthetic_data(2000)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Hypertension rate: {df['Hypertension'].mean():.2%}")
    
    # Separate features and target
    X = df.drop('Hypertension', axis=1)
    y = df['Hypertension']
    
    # Define categorical and numerical columns
    categorical_columns = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
    numerical_columns = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']
    
    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )
    
    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Training model...")
    model.fit(X, y)
    
    # Test the model
    train_score = model.score(X, y)
    print(f"Training accuracy: {train_score:.3f}")
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), 'hypertension_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model

if __name__ == "__main__":
    model = train_model()
    print("Model training completed successfully!")


