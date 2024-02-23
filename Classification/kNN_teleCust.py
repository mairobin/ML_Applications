import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Find Customer Category for Customers with some features like age, income etc.
df = pd.read_csv('../Datasets/teleCust1000t.csv')

# How many values for categories are there?
customers_per_category = df['custcat'].value_counts()

# Tip: explore data with visualizations like histograms or scatterplots
features = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
#plt.show()



# Normalize Data -> here Standardization

normalized_features = preprocessing.StandardScaler().fit(features).transform(features.astype(float))
labeled = df['custcat'].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( normalized_features, labeled, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k = 4
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

yhat = neigh.predict(X_test)
yhat[0:5]

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
#plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
#plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()




"""







# Train / Test Split
from sklearn.model_selection import train_test_split
train_input, test_input, train_labels, test_labels = train_test_split(normalized_features, labeled, test_size=0.2, random_state=4)
print('Train set:', train_input.shape, train_labels.shape) # y-Values are categories
print('Test set:', test_input.shape, test_labels.shape)

from sklearn.neighbors import KNeighborsClassifier
k = 3
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(train_input, train_labels)
res = neigh.predict(test_input)
res[0:5]

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(train_labels, neigh.predict(train_input)))
print("Test set Accuracy: ", metrics.accuracy_score(test_labels, res))


### How to find right k-value -> Try and Plot

Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(train_input, train_labels)
    predicted_labels = neigh.predict(test_input)
    mean_acc[n - 1] = metrics.accuracy_score(test_input, predicted_labels)
    std_acc[n - 1] = np.std(predicted_labels == test_labels) / np.sqrt(predicted_labels.shape[0])

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

print()
"""

