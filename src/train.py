# 1. IMPORT LIBRARIES
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

print("--- LIBRARIES IMPORTED ---")

# 2. LOAD AND CLEAN DATA
# The path starts with 'data/' because we run the script from the main 'churn_mlops' folder.
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Simple data cleaning: convert 'TotalCharges' to numbers, drop rows with missing values.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

print("--- DATA LOADED AND CLEANED ---")

# 3. DEFINE FEATURES AND TARGET
# For simplicity, we'll use only three numerical features.
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features]
# The 'Churn' column is our target. We convert 'Yes'/'No' to 1/0.
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

print("--- FEATURES AND TARGET DEFINED ---")

# 4. SPLIT DATA
# We split the data: 80% for training the model, 20% for testing its performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- DATA SPLIT COMPLETE ---")

# 5. START MLFLOW EXPERIMENT
# This tells MLflow to group our runs under this experiment name.
mlflow.set_experiment("Churn_Prediction_Experiment")

# 'with mlflow.start_run()' logs everything inside this block to one specific run.
with mlflow.start_run():
    print("--- MLFLOW RUN STARTED ---")
    
    # 6. TRAIN THE MODEL
    # We use XGBoost, a powerful and popular algorithm for this type of problem.
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    print("--- MODEL TRAINING COMPLETE ---")

    # 7. EVALUATE AND LOG
    # We use the trained model to make predictions on the test data it has never seen.
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    # Log the accuracy metric to MLflow so we can see it in the dashboard.
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model trained with accuracy: {accuracy}")
    
    # 8. SAVE THE MODEL
    # We save the final trained model into the 'models' folder.
    # The API we build later will load this specific file to make predictions.
    os.makedirs("models", exist_ok=True)
    model.save_model("models/churn_model.xgb")
    print("--- MODEL SAVED TO 'models/churn_model.xgb' ---")

print("--- SCRIPT FINISHED ---")