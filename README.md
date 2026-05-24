# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load the employee dataset and use One-Hot Encoding (get_dummies) to convert categorical text into numbers so the model can process them.
2. Dataset Splitting: Separate the data into features (X) and the target label (y), then split them into Training (80%) and Testing (20%) sets.
3. Model Training: Initialize a DecisionTreeClassifier and "fit" it to the training data, allowing the algorithm to learn patterns and decision rules.
4. Evaluation & Visualization: Predict outcomes for the test set to calculate the Accuracy Score, and generate a visual flowchart of the tree using plot_tree.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sridharan B
RegisterNumber:  212225230272
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
data = pd.read_csv("Employee.csv")
data = pd.get_dummies(data, drop_first=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.show()
```

## Output:

<img width="1020" height="530" alt="image" src="https://github.com/user-attachments/assets/e6a88dab-8c90-4222-b01c-e993ae7e160c" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
