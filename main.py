import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("churn.csv")

# Display first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Dataset shape
print("\nDATASET SHAPE:")
print(df.shape)

# Dataset information
print("\nDATASET INFORMATION:")
print(df.info())

# Statistical description
print("\nSTATISTICAL DESCRIPTION:")
print(df.describe())

# Column names
print("\nCOLUMNS:")
print(df.columns.tolist())

# Missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Duplicate records
print("\nDUPLICATES:")
print(df.duplicated().sum())

print("\nCHURN DISTRIBUTION:")
print(df["Churn"].value_counts())

print("\nCHURN PERCENTAGE:")
print(df["Churn"].value_counts(normalize=True) * 100)

#EDA
#Target visualization
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

#Gender vs Churn
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="gender", hue="Churn")
plt.title("Gender vs Churn")
plt.show()

#Contract vs Churn
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.show()

#Monthly Charges distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="MonthlyCharges", kde=True)
plt.title("Monthly Charges Distribution")
plt.show()

#Tenure vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs Churn")
plt.show()

# Contract vs Churn
print(pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100)

#internet service vs churn
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="InternetService", hue="Churn")
plt.title("Internet Service vs Churn")
plt.show()

#Payment Method vs Churn
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="PaymentMethod", hue="Churn")
plt.title("Payment Method vs Churn")
plt.xticks(rotation=25)
plt.show()

#Monthly Charges vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.show()

#Tenure vs Churn
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    kde=True,
    bins=30
)
plt.title("Tenure Distribution by Churn")
plt.show()

#Correlation Analysis
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()

#Data Cleaning & Feature Engineering
#Check Data Types
print("Data Types:")
print(df.dtypes)

#Handle Missing Values
print("Missing values before cleaning:")
print(df.isnull().sum())
print("\nBlank TotalCharges:")
print((df["TotalCharges"].astype(str).str.strip() == "").sum())
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
#Missing values handle karo
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print(df.isnull().sum())

#Remove Duplicate Records
print("Duplicates before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates after:", df.duplicated().sum())

#Remove Unnecessary Column
df = df.drop_duplicates()
df = df.drop(columns=["customerID"])

#Feature Engineering
#tenure Groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Medium", "Long", "Very Long"]
)
# monthly Charges Category
df["ChargesCategory"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 70, 120],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

#final dataset after cleaning and feature engineering
print("Final Shape:", df.shape)

print("\nFinal Data Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Rows:")
print(df.head())

#Feature Selection, Encoding & Scaling
#Separate Features and Target

# Target variable
y = df["Churn"].map({"No": 0, "Yes": 1})

# Input features
X = df.drop(columns=["Churn"])

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

#. Identify Numerical and Categorical Columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Numerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

#Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

#Create Preprocessing Pipelines
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessing pipeline created successfully!")

# Logistic Regression and Evaluation Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Create and train Logistic Regression model
log_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train
log_model.fit(X_train, y_train)
print("Logistic Regression Model Trained Successfully!")

# Predict classes
y_pred = log_model.predict(X_test)

# Predict churn probabilities
y_prob = log_model.predict_proba(X_test)[:, 1]

print("Predictions:", y_pred[:10])
print("Probabilities:", y_prob[:10])

print("\n----- MODEL EVALUATION -----")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
))

# Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
#Create Multiple Models
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


results = []

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred_model = pipeline.predict(X_test)
    y_prob_model = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred_model),
        "Precision": precision_score(y_test, y_pred_model),
        "Recall": recall_score(y_test, y_pred_model),
        "F1 Score": f1_score(y_test, y_pred_model),
        "ROC-AUC": roc_auc_score(y_test, y_prob_model)
    })

    print(name, "trained successfully!")

#Compare Model
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)
print("\nMODEL COMPARISON:")
print(results_df.round(4))

#Visualize the Comparison
plt.figure(figsize=(10, 6))

results_melted = results_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

sns.barplot(
    data=results_melted,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Machine Learning Model Comparison")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# create a random forest pipeline
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

# define hyperparameter grid
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, 15, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

#train the model using GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

#Find Best Parameters
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest CV F1 Score:")
print(grid_search.best_score_)

#Evaluate Tuned Model
best_model = grid_search.best_estimator_

y_pred_tuned = best_model.predict(X_test)
y_prob_tuned = best_model.predict_proba(X_test)[:, 1]

print("\nTUNED RANDOM FOREST RESULTS")

print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Precision:", precision_score(y_test, y_pred_tuned))
print("Recall:", recall_score(y_test, y_pred_tuned))
print("F1 Score:", f1_score(y_test, y_pred_tuned))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_tuned))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_tuned,
    target_names=["No Churn", "Churn"]
))

#Get Prediction Probabilities
y_prob = best_model.predict_proba(X_test)[:, 1]

#Test Different Thresholds
from sklearn.metrics import precision_score, recall_score, f1_score
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
threshold_results = []
for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(y_test, y_pred_threshold),
        "Recall": recall_score(y_test, y_pred_threshold),
        "F1 Score": f1_score(y_test, y_pred_threshold)
    })

threshold_df = pd.DataFrame(threshold_results)
print(threshold_df.round(4))

#Visualize Threshold Performance
plt.figure(figsize=(9, 5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)
plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)
plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1 Score"],
    marker="o",
    label="F1 Score"
)
plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization")
plt.legend()
plt.grid()
plt.show()

#Select the Best Threshold
best_threshold = threshold_df.loc[
    threshold_df["F1 Score"].idxmax(),
    "Threshold"
]

print("Best Threshold:", best_threshold)

#import joblib
import joblib

#Save the Model
joblib.dump(best_model, "customer_churn_model.pkl")
print("Model saved successfully!")
# save the threshold
joblib.dump(best_threshold, "threshold.pkl")
print("Threshold saved successfully!")

#Save the Feature Names
joblib.dump(X_train.columns.tolist(), "feature_names.pkl")
print("Feature names saved successfully!")

#Test Whether the Model Loads
loaded_model = joblib.load("customer_churn_model.pkl")
print("Model loaded successfully!")
predictions = loaded_model.predict(X_test)
print("Sample Predictions:", predictions[:10])

