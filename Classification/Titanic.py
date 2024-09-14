import pandas as pd

df = pd.read_csv("../Datasets/Titanic_prediction.csv")

#describe = df.describe()
#info = df.info()

#df_survived = df[df['Survived'] == 1]

#describe_survived = df_survived.describe()


# Data Cleansing: Data Types

# Get unique data types in the feature column
# Do it for features with Object Datatype

cols_dtype_object = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
for col in cols_dtype_object:
    unique_data_types = df[col].apply(type).unique()
    print(f"{col}, DTypes{unique_data_types}")

"""
Name, DTypes[<class 'str'>]
Sex, DTypes[<class 'str'>]
Ticket, DTypes[<class 'str'>]
Cabin, DTypes[<class 'float'> <class 'str'>]
Embarked, DTypes[<class 'str'> <class 'float'>]
"""

"""
DTypes[<class 'float'> <class 'str'>], it typically indicates that the column contains a mix of float and string values. 
This can happen if the column contains both numerical and textual data, or if missing values are represented as floats (NaN) and non-missing values are strings.
"""

# Deal with NaNs

# Options: Replace with Placeholder, Drop rows, Do nothing, Imputation
# Avoid dropping rows as the dataset is small anyway
#
# For String Columns

# Cabin: Too many Categories. Too much granularity1-N Relationship Multiple Persons for one Cabine
# Do i need to split cabine ID further, maybe into zones of the ship? -> lower granularity
df['Cabin'] = df['Cabin'].fillna('UNKOWN')

# Categorial -> only 3 Options
unique_values = df['Embarked'].unique()
df['Embarked'] = df['Embarked'].fillna('UNKOWN')

cols_dtype_object = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
for col in cols_dtype_object:
    unique_data_types = df[col].apply(type).unique()
    print(f"{col}, DTypes{unique_data_types}")

# DONE!


### Lets have a deeper look at the Cabine Column

#Cabin seems to have two parts Letter and Number
#Cabin Column is non Atomic

# Get it atomic
# Split the values in the Cabin column
cabines_separated = df['Cabin'].str.split(expand=True)

# See if first char is always the same -> Most time the letter is the same.
zones_separated = cabines_separated.apply(lambda col: col.str[0])

#Separate letters from number -> Start with one Column
# Need simplification for problem solving

# For one Column
#split_columns[['Letters', 'Numbers']] = split_columns['1'].str.split(r'(\d+)', expand=True)
### ->  This gets to messy



# use OneHotEncoding "HasCabinOnZone_X"

# What zones are there?

all_zones = set()
for col in zones_separated.columns:
    zones_for_col = zones_separated[col].dropna().unique()
    all_zones.update(zones_for_col)


all_zones_ordered = sorted(list(all_zones))

# Split Values in the Cabin column, but without expanding

df['Cabin'] = df['Cabin'].str.split()

# Create new columns for each unique letter
for zone in all_zones_ordered:
    df[f'CabinZone{zone}'] = df['Cabin'].apply(lambda x: any(element.startswith(zone) for element in x))

# Drop the old Cabin Column
df = df.drop('Cabin', axis=1)

# Drop the passenger ID column as we have own indices
df = df.drop('PassengerId', axis=1)

# For Tickets Column maybe there is also a pattern that could be interesting, but no initial guess so leave it for later


# Now the easy preprocessing

# For Sex, Embarked use OneHotEncoding
# Perform one-hot encoding for the 'Sex' column
one_hot = pd.get_dummies(df['Sex'])
df = pd.concat([one_hot, df], axis=1)

one_hot = pd.get_dummies(df['Embarked'])
one_hot = one_hot.drop(one_hot.columns[-1], axis=1)
df = pd.concat([one_hot, df], axis=1)

df = df.drop('Sex', axis=1)
df = df.drop('Embarked', axis=1)
df = df.drop('Name', axis=1)
df = df.drop('Ticket', axis=1)


###
#Model Selection
####

"""
We have a tiny dataset
Binary Classification
Categorial  and numerical data 
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'df' is your pandas DataFrame with features and target variable
# Splitting the DataFrame into features (X) and target variable (y)
target_variable = 'Survived'
X = df.drop(target_variable, axis=1)  # Drop the target variable column
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report for more detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Lets do a decision Tree

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Assuming 'df' is your pandas DataFrame with features and target variable
# Splitting the DataFrame into features (X) and target variable (y)
target_variable = 'Survived'
X = df.drop(target_variable, axis=1)  # Drop the target variable column
y = df[target_variable]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the DecisionTreeClassifier
#clf = DecisionTreeClassifier(random_state=42)

# Max depth of 5 leads to higher accuracy
clf = DecisionTreeClassifier(random_state=42, max_depth=3)

# Training the Decision Tree classifier
clf.fit(X_train, y_train)

# Making predictions on the test set
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Convert clf.classes_ to strings
class_names = [str(cls) for cls in clf.classes_]

# Visualize the decision tree
plt.figure()
plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True)
plt.show()


# Interpretation

# Left Branch -> Condition True
# Right Branch -> Condition not True

#value = [DEAD, SURVIVED]

# Think of Impurity boxes with balls of different colors
# We want the split that reduces impurity the most


# Focus on the left
# First split on male -> see how the ratio D/S changes on the right side
# CabineU: when you were not in CabineU the Chances where good to survive
# PClass under 2.5 (1 or 2) is an estimator for survival -> when 1 is highest then its not surprising


"""
Decison Tree with max Depth against overfitting is a good model to check the data.
However keep in mind:

Overfitting: Decision trees are prone to overfitting, especially when the tree depth is not properly controlled. Techniques like pruning or using ensemble methods like Random Forests can help mitigate this issue.

Instability: Small variations in the data can lead to significantly different decision trees. Ensemble methods like Random Forests or Gradient Boosting can help improve stability.

Bias: Decision trees can be biased towards features with a large number of levels or categories.

Limited Expressiveness: Decision trees may not capture complex relationships in the data as effectively as more sophisticated models like neural networks.
"""



#### Conclusion

# Ich habe es mir sehr leicht gemacht hinsicht des One Hot Encodings.
# Beispiel Zonen: jede eine einzelne Spalte im Model war nur die Zone U
# Die Dimensionalität haben wir stark erhöht.




print()