import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import learning_curve,validation_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_LEAF
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve


import sys
import time


def decision_tree(X_train, X_test, y_train, y_test, depth, threshold, title):
    start_time = time.time()  # needed to estimate the runtime


    # form and train the default classifier and predict
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
    # traverse the tree and remove all children of the nodes with minimum
    # class count less than threshold (the last parameter)
    prune_dt(dt.tree_, 0, threshold)
    predict = dt.predict(X_test)

    # print accuracy score and runtime
    accuracy = accuracy_score(y_test, predict)
    print("This is the accuracy of decision tree ", accuracy)
    print("--- %s seconds ---" % (time.time() - start_time))

    # plot the training / testing curve

    cv = KFold(n_splits=5)
    plot_learning_curve(dt, title, X_train, y_train, None, cv=cv, n_jobs=4)

    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,
                                                            n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing score")

    plt.legend(loc="best")
    return plt

def prune_dt(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF

    if inner_tree.children_left[index] != TREE_LEAF:
        prune_dt(inner_tree, inner_tree.children_left[index], threshold)
        prune_dt(inner_tree, inner_tree.children_right[index], threshold)



if __name__ == "__main__":
	dataType = sys.argv[1:]
	if (dataType[0] == "car"):
		data=pd.read_excel('car_data1.xlsx')
		le=LabelEncoder()
		for i in data.columns:
			data[i]=le.fit_transform(data[i])

		X=data[data.columns[:-1]]
		y=data['class']

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
		decision_tree(X_train, X_test, y_train, y_test, 7, 3, "Decision Tree on Car Evaluation Set")
	elif (dataType[0] == "cancer"):
		dataset = pd.read_csv('cancer_data.csv')
		X = dataset.iloc[:, 2:32].values
		Y = dataset.iloc[:, 1].values

		labelencoder_Y = LabelEncoder()
		y = labelencoder_Y.fit_transform(Y)


		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
		decision_tree(X_train, X_test, y_train, y_test, 2, 20, "Decision Tree on Breast Cancer Set")