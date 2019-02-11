import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import learning_curve,validation_curve

from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import sys
import time

def svm(X_train, X_test, y_train, y_test, title):
	start_time = time.time()

	# svc = SVC()
	svc = SVC(kernel='linear')
	# svc = SVC(kernel="poly")

	svc.fit(X_train, y_train)
	predict = svc.predict(X_test)
	accuracy = accuracy_score(y_test, predict)
	print("This is the accuracy of boosting ", accuracy)
	print("--- %s seconds ---" % (time.time() - start_time))

	cv = KFold(n_splits=5)
	plot_learning_curve(svc, title, X_train, y_train.ravel(), None, cv=cv, n_jobs=4)
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

if __name__ == "__main__":
	dataType = sys.argv[1:]
	if (dataType[0] == "car"):
		data=pd.read_excel('car_data1.xlsx')
		le=LabelEncoder()
		for i in data.columns:
			data[i]=le.fit_transform(data[i])

		X=data[data.columns[:-1]]
		y=data['class']
		scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
		X = scaler.fit_transform(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
		svm(X_train, X_test, y_train, y_test, "SVM for Car Evaluation Set")

	elif (dataType[0] == "cancer"):
		dataset = pd.read_csv('cancer_data.csv')
		X = dataset.iloc[:, 2:32].values
		Y = dataset.iloc[:, 1].values

		labelencoder_Y = LabelEncoder()
		y = labelencoder_Y.fit_transform(Y)

		scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
		X = scaler.fit_transform(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
		svm(X_train, X_test, y_train, y_test, "SVM for Breast Cancer Set")