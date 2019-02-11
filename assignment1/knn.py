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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


import sys
import time


def knn(X_train, X_test, y_train, y_test): 

	start_time = time.time()
	list_x = []
	list_y = []
	list_train = []
	max_v = 0
	max_i = ()
	my_list = []
	for i in range(1, 50):
		knn_clf = KNeighborsClassifier(n_neighbors=i)
		knn_clf.fit(X_train, y_train)
		predict = knn_clf.predict(X_test)
		accuracy = accuracy_score(y_test, predict)
		predict_train = knn_clf.predict(X_train)
		accuracy_train = accuracy_score(y_train, predict_train)
		# print(i, "this is the accuracy ", accuracy)
		list_x.append(i)
		list_y.append(accuracy)
		list_train.append(accuracy_train)
		if max_v < accuracy:
			max_v = accuracy
			max_i = i
	list_x = np.array(list_x)
	list_y = np.array(list_y)
	print ("k", max_i)
	print ("accuracy", max_v)
	print("--- %s seconds ---" % (time.time() - start_time))

	plt.figure()
	plt.grid()
	# plt.plot(range(2,30),avg_score)
	plt.title("car evaluation: change of accuracy by number of neighbors")
	plt.xlabel("K (number of neighbors)")
	plt.ylabel("Accuracy")
	plt.plot(list_x, list_y, 'r', label="test")
	plt.plot(list_x, list_train, 'b', label="training")
	plt.show()
	

if __name__ == "__main__":
	dataType = sys.argv[1:]
	if (dataType[0] == "car"):
		data=pd.read_excel('car_data1.xlsx')
		le=LabelEncoder()
		for i in data.columns:
			data[i]=le.fit_transform(data[i])

		X=data[data.columns[:-1]]
		y=data['class']

		# scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
		# X = scaler.fit_transform(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)

		knn(X_train, X_test, y_train, y_test)
	elif (dataType[0] == "cancer"):
		dataset = pd.read_csv('cancer_data.csv')
		X = dataset.iloc[:, 2:32].values
		Y = dataset.iloc[:, 1].values

		labelencoder_Y = LabelEncoder()
		y = labelencoder_Y.fit_transform(Y)

		scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
		X = scaler.fit_transform(X)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

		# scaler = StandardScaler()
		# scaler.fit(X_train)
		# X_train = scaler.transform(X_train)
		# X_test = scaler.transform(X_test)

		knn(X_train, X_test, y_train, y_test)

