import pandas as pd

# Section 1: Data Collection
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Section 2: Data Cleaning and Preparation
# Part A: Handle Missing Values
print("Missing values before handling:")
print(df.isnull().sum())

# Part B: Drop Irrelevant Features

# Part C: Convert Categorical Variables


# Section 3: Exploratory Data Analysis (EDA)

# Section 4: Splitting the Data

# Section 5: Model Training
# Part A: Baseline Model
# Part B: Model Evaluation

# Section 6: Hyperparameter Tuning

# Section 7: Final Model Evaluation
# Part A: Retrain with Best Parameters
# Part B: Evaluate on Test Set

# Section 8: Reporting
# Part A: Summary of Findings
# Part B: Visualizations
# Part C: Challenges and Improvements

