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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import sys
import time

def nn(X_train, X_test, y_train, y_test, title, h_title):
    start_time = time.time()

    list_x = []
    list_y = []
    list_train = []
    max_v = 0
    max_i = ()
    my_list = []
    units = 15
    my_list.append(units)
    for i in range(1, 40):
        layer = tuple(my_list)
        mlp = MLPClassifier(hidden_layer_sizes=layer, max_iter=5000)
        mlp.fit(X_train, y_train.ravel())
        predict = mlp.predict(X_test)
        predict_train = mlp.predict(X_train)
        accuracy = accuracy_score(y_test, predict)
        accuracy_train = accuracy_score(y_train, predict_train)
        # print(i, "this is the accuracy ", accuracy)
        list_x.append(i)
        list_y.append(accuracy)
        list_train.append(accuracy_train)
        if max_v < accuracy:
            max_v = accuracy
            max_i = layer
        my_list.append(units)
    list_x = np.array(list_x)
    list_y = np.array(list_y)

    # mlp = MLPClassifier(hidden_layer_sizes=(15, 15, 15, 15), max_iter=5000)
    # mlp.fit(X_train, y_train.ravel())
    # predict = mlp.predict(X_test)
    # accuracy = accuracy_score(y_test, predict)
    print(max_v, max_i)
    
    plt.figure()
    plt.grid()
    plt.title(h_title)
    plt.xlabel("Number of hidden layers")
    plt.ylabel("Accuracy")
    plt.plot(list_x, list_y, 'r', label="test")
    plt.plot(list_x, list_train, 'b', label="training")
    plt.show()

    print("This is the accuracy of neural network ", accuracy)
    print("--- %s seconds ---" % (time.time() - start_time))

    # cv = KFold(n_splits=5)
    # plot_learning_curve(mlp, title, X_train, y_train.ravel(), None, cv=cv, n_jobs=4)
    # plt.show()

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

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        nn(X_train, X_test, y_train, y_test, "Neural Networks for Car Evaluation Set", "Car Evaluation : Change of accuracy by number of hidden layer")

    elif (dataType[0] == "cancer"):
        dataset = pd.read_csv('cancer_data.csv')
        X = dataset.iloc[:, 2:32].values
        Y = dataset.iloc[:, 1].values

        labelencoder_Y = LabelEncoder()
        y = labelencoder_Y.fit_transform(Y)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        nn(X_train, X_test, y_train, y_test, "Neural Networks for Breast Cancer Set", "Breast Cancer : Change of accuracy by number of hidden layer")

