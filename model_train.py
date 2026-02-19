import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("Step 1: Loading Dataset...")
# Load the dataset from the same directory
df = pd.read_csv('water_potability.csv')

print("Step 2: Handling Missing Values & Outliers...")
# 1. Fill missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

# 2. Function to cap outliers using IQR
def cap_outliers(dataframe, col_name):
    Q1 = dataframe[col_name].quantile(0.25)
    Q3 = dataframe[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap values outside the bounds
    dataframe[col_name] = np.where(dataframe[col_name] > upper_bound, upper_bound,
                                   np.where(dataframe[col_name] < lower_bound, lower_bound, dataframe[col_name]))
    return dataframe

# Apply capping to all feature columns
feature_cols = [col for col in df.columns if col != 'Potability']
for col in feature_cols:
    df = cap_outliers(df, col)

print("Step 3: Feature Engineering & Encoding...")
# 1. Create 'ph_group'
def classify_ph(ph_val):
    if ph_val < 6.5: return 'Acidic'
    elif ph_val > 8.5: return 'Alkaline'
    else: return 'Neutral'

df['ph_group'] = df['ph'].apply(classify_ph)

# 2. One-Hot Encoding for 'ph_group'
df_encoded = pd.get_dummies(df, columns=['ph_group'], prefix='ph', dtype=int)

print("Step 4: Data Splitting...")
# Separate Features (X) and Target (y)
X = df_encoded.drop('Potability', axis=1)
y = df_encoded['Potability']

# Train-Test Split (80% Train, 20% Test)
# Note: We do NOT scale here. The pipeline will handle it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Step 5: Creating & Training the Pipeline...")
# Create a Pipeline with StandardScaler and the Best RandomForest Model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=173,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=2,
        bootstrap=True,
        random_state=42
    ))
])

# Train the ENTIRE pipeline on the unscaled training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Pipeline Training Completed! Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

print("Step 6: Saving the Pipeline...")
# Save the entire pipeline (Scaler + Model) as a single pickle file
with open('water_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("Success! 'water_pipeline.pkl' has been saved.")