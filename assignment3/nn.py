import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
# import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pandas.read_csv("ccf_PCA_KM.csv")

X = df.drop(columns = ['class'])
Y = df['class']

from sklearn.neural_network import MLPClassifier

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 29)
7
clf = MLPClassifier(solver = 'adam', alpha = 0.3, hidden_layer_sizes = (40, 10), random_state = 1)
clf.fit(X_train, Y_train.values.ravel())
print(accuracy_score(clf.predict(X_test), Y_test))

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=clf,
                                                       X=X_train,
                                                       y=Y_train,
                                                       train_sizes=np.linspace(0.1, 1.0, 20),
                                                       cv=10)

# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)

# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')

plt.title('Original Credit')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()