import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Section 1: Data Collection
df = pd.read_csv("titanic.csv")


# Section 2: Data Cleaning and Preparation
# Part A: Handle Missing Values
print("Missing values before handling:")
print(df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Part B: Drop Irrelevant Features
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Part C: Convert Categorical Variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


# Section 3: Exploratory Data Analysis (EDA)


# Section 4: Splitting the Data
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)


# Section 5: Model Training
# Part A: Baseline Model
x_train = df_train.drop('Survived', axis=1)
y_train = df_train['Survived']
x_val = df_val.drop('Survived', axis=1)
y_val = df_val['Survived']
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Part B: Model Evaluation
y_pred = rf.predict_proba(x_train)[:, 1]
print("ROC-AUC on Training Set:")
print(roc_auc_score(y_train, y_pred))

# y_pred = rf.predict_proba(x_train)[:, 1]
print("ROC-AUC on Validation Set:")
print(roc_auc_score(y_val, y_pred))

# Section 6: Hyperparameter Tuning


# Section 7: Final Model Evaluation
# Part A: Retrain with Best Parameters
# Part B: Evaluate on Test Set


# Section 8: Reporting
# Part A: Summary of Findings
# Part B: Visualizations
# Part C: Challenges and Improvements

