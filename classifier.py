import pandas as pd
import numpy as np
#Librerías necesarias para realizar el proyecto
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Librerías necesarias para los clasificadores
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier #Kstar
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
#https://aprendeconejemplos.org/python/preprocesamiento-de-datasets-con-scikit-learn-y-pandas Grax a este chulo
#https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
def main():
	data = pd.read_csv("wine.csv")
	columns = data.columns.to_list()[0:]
	# print(columns)
	x = data[columns[:-1]].values
	y = data[columns[-1]].values
	#Normalizando los datos
	min_max_scaler = MinMaxScaler()
	X = min_max_scaler.fit_transform(x)

	#Preparando los datos
	X_train, X_test, y_train, y_test = X, X, y, y

	#Decision Tree
	DecisionTree = DecisionTreeClassifier()
	DecisionTree.fit(X_train, y_train)
	y_pred = DecisionTree.predict(X_test)
	print("Decision Tree:")
	print(("Accuracy: ", accuracy_score(y_test, y_pred)))
	print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))
	
	#Naive Bayes
	naive_bayes = GaussianNB()
	naive_bayes.fit(X_train, y_train)
	y_pred = naive_bayes.predict(X_test)
	print("\nNaive Bayes")
	print("Accuracy: ", accuracy_score(y_test, y_pred))
	print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

	#Logistic
	Logistic = LogisticRegression()
	Logistic.fit(X_train, y_train)
	y_pred = Logistic.predict(X_test)
	print("\nLogistic:")
	print(("Accuracy: ", accuracy_score(y_test, y_pred)))
	print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

	#LinearSVC
	Linear = LinearSVC()
	Linear.fit(X_train, y_train)
	y_pred = Linear.predict(X_test)
	print("\nLinear:")
	print(("Accuracy: ", accuracy_score(y_test, y_pred)))
	print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

	#MLPClassifier
	multi = MLPClassifier()
	multi.fit(X_train,y_train)
	y_pred = multi.predict(X_test)
	print("\nMLPClassifier:")
	print(("Accuracy: ", accuracy_score(y_test, y_pred)))
	print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

	#Support vector classification
	svc = SVC()
	svc.fit(X_train,y_train)
	y_pred = svc.predict(X_test)
	print("\nSupport vector classification:")
	print(("Accuracy: ", accuracy_score(y_test, y_pred)))
	print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

	#Random Forest Classifier
	R_Forest = RandomForestClassifier()
	R_Forest.fit(X_train,y_train)
	y_pred = R_Forest.predict(X_test)
	print("\nRandom Forest Classifier:")
	print(("Accuracy: ", accuracy_score(y_test, y_pred)))
	print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

	#KNeighborsClassifier
	kdtree = KNeighborsClassifier()
	kdtree.fit(X_train,y_train)
	y_pred = kdtree.predict(X_test)
	print("\nKNeighborsClassifier:")
	print(("Accuracy: ", accuracy_score(y_test, y_pred)))
	print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

	#MultinomialNB
	Bayesmulti = MultinomialNB()
	Bayesmulti.fit(X_train,y_train)
	y_pred = Bayesmulti.predict(X_test)
	print("\nMultinomialNB:")
	print(("Accuracy: ", accuracy_score(y_test, y_pred)))
	print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))

	#GaussianProcessRegressor


	#OneVsRestClassifier
	restclas = OneVsRestClassifier(LinearSVC(random_state=0))
	restclas.fit(X_train,y_train)
	y_pred = restclas.predict(X_test)
	print("\nOneVsRestClassifier:")
	print(("Accuracy: ", accuracy_score(y_test, y_pred)))
	print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))	
main()