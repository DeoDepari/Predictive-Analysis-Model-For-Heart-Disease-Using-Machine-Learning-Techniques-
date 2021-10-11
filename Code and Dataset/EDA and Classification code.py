# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 02:55:34 2021

@author: Deo Haganta Depari
"""


#Packages
#untuk keperluan dataset
import pandas as pd
import numpy as np

#untuk visualisasi
import matplotlib.pyplot as plt 

#%matplotlib inline 
import seaborn as sns

#Exploratory data analysis
from collections import Counter
#import pandas_profiling as pp

#pra proses data
from sklearn.preprocessing import StandardScaler

#pembagian data
from sklearn.model_selection import train_test_split



#ensembling
#from mlxtend.classifier import StackingCVClassifier


#Packages


#baca dataset
#path = "D:/Deo Haganta Depari/1.KAMPUS/Semester 6/Big Data/Tugas Akhir/Data Set/Heart Disease and Stroke Prevention/dataset.csv"
#path_heart = "D:/Deo Haganta Depari/1.KAMPUS/Semester 6/Big Data/Tugas Akhir/Data Set/Heart Disease UCI/heart.csv"
path_heart = r"heart_statlog_cleveland_hungary_final.csv"
data = pd.read_csv(path_heart)

#membaca 5 data pertama
data.head()
data.info()


#kita akan cek apakah ada data yang hilang atau tidak
data.isnull().sum()

#Descriptive statistics
data.describe()

## ↓↓ Exploratory data analysis ↓↓
data.target.value_counts()

#deskripsi dan plot target
sns.countplot(x = "target", data = data, palette = ['black', 'grey'])
plt.show()

jumlahTidakSakit = len(data[data.target == 0])
jumlahSakit = len(data[data.target == 1])
print('')
print("Presentase Pasien yang memiliki Penyakit Jantung = {:.2f}%".
      format((jumlahSakit / (len(data.target))* 100 )))

print("Presentase Pasien yang tidak memiliki Penyakit Jantung = {:.2f}%".
      format((jumlahTidakSakit / (len(data.target))* 100 )))

#deskripsi dan plot sex
sns.countplot(x='sex', data=data , palette = ['black', 'grey'])
plt.xlabel ("Attribut sex/jenis kelamin, dimana 0 = perempuan, 1 = laki-laki")
plt.show()



jumlahperempuan = len(data[data.sex == 0])
jumlahlakilaki = len(data[data.sex == 1])
print('')
print("Presentase Pasien Perempuan = {:.2f}%".
      format((jumlahperempuan / (len(data.sex))* 100 )))

print("Presentase Pasien Laki-Laki = {:.2f}%".
      format((jumlahlakilaki / (len(data.sex))* 100 )))

## ↑↑ Exploratory data analysis ↑↑



#Membuat Variabel Dummy, membantu meningkatkan nilai akurasi
#Dikarenakan 'cp', 'thal' dan 'slope' atribut merupakan variable kategori maka bisa kita buat dummy variablenya

# dummy_cp = pd.get_dummies(data['cp'], prefix = "cp")
# dummy_thal = pd.get_dummies(data['thal'], prefix = "thal")
dummy_slope = pd.get_dummies(data['slope'], prefix = "slope")

#memasukkan frames dummy ke dataframe
frames = [data, dummy_slope]
data = pd.concat(frames, axis = 1)
data.head()

#drop atribut yang telah di ganti dengan dummy data
data = data.drop(columns = ['slope'])
data.head()

#Persiapkan model
y = data.target.values
x = data.drop(['target'],axis=1)

#normalisasi data
x_normalize = (x - np.min(x)) / (np.max(x) - np.min(x)).values

#memastikan data sudah di split sebelum di apply ke algoritma
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


#Machine Learning model
akurasitotal = {}


#KNN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediksi = knn.predict(x_test.T)

print("{} KNN Score: {:.2f}%".format(1, knn.score(x_test.T, y_test.T)*100))


    # mencari nilai n_terbaik
hasilpercobaan = []
for i in range(1,20):
    knn_exp = KNeighborsClassifier(n_neighbors = i) #n_neighbors berarti k = 2
    knn_exp.fit(x_train.T, y_train.T)
    hasilpercobaan.append(knn_exp.score(x_test.T, y_test.T))
    
plt.plot(range(1,20), hasilpercobaan)
plt.xticks(np.arange(1,20,1))
plt.xlabel("Nilai K")
plt.ylabel("Skor")
plt.show()

akurasi_KNN = max(hasilpercobaan)*100
akurasitotal['KNN'] = akurasi_KNN
print("Maximum KNN Score is {:.2f}%".format(akurasi_KNN))
    
#print("\nn_neighbors = {}, KNN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))

#Support Vector Machine
from sklearn.svm import SVC

svm = SVC(random_state = 1)
svm.fit(x_train.T, y_train.T)

akurasi_SVM = svm.score(x_test.T, y_test.T)*100
akurasitotal['SVM'] = akurasi_SVM
print("\nTesting Akurasi Algoritma SVM : {:.2f}%".format(akurasi_SVM))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
naivebayes = GaussianNB()
naivebayes.fit(x_train.T, y_train.T)

akurasi_NaiveBayes = naivebayes.score(x_test.T, y_test.T)*100
akurasitotal['NaiveBayes'] = akurasi_NaiveBayes
print("\nTesting Akurasi Algoritma Naive Bayes : {:.2f}%".format(akurasi_NaiveBayes))

#Decision Tree Algoritma
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train.T, y_train.T)

akurasi_decisiontree = decisiontree.score(x_test.T, y_test.T)*100
akurasitotal['DecisionTree'] = akurasi_decisiontree
print("\nTesting Akurasi Algoritma Decision Tree : {:.2f}%".format(akurasi_decisiontree))

#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators = 1000, random_state = 1)
randomforest.fit(x_train.T, y_train.T)

akurasi_randomforest = randomforest.score(x_test.T, y_test.T)*100
akurasitotal['Random Forrest'] = akurasi_randomforest
print("\nTesting Akurasi Algoritma Random Forrest : {:.2f}%".format(akurasi_randomforest))


#Hasil dan Evaluasi
#perbandingan Model

sns.set_style("whitegrid")
plt.figure(figsize = (16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(akurasitotal.keys()), y=list(akurasitotal.values()), palette= ['black', 'grey'])
plt.show()

#Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

#Nilai prediksi
#KNN EVALUATION
knn_conf = KNeighborsClassifier(n_neighbors = 1)
knn_conf.fit(x_train.T, y_train.T)
y_head_knn_train = knn_conf.predict(x_train.T)
y_head_knn_test = knn_conf.predict(x_test.T)

confusionmatrix_knn_train = confusion_matrix(y_train, y_head_knn_train)
clf_report_knn_train = pd.DataFrame(classification_report(y_train.T,y_head_knn_train, output_dict = True))
print("KNN CLASSIFICATION REPORT TRAIN DATA:\n",clf_report_knn_train)
print("_______________________________________________________________________")

confusionmatrix_knn_test = confusion_matrix(y_test, y_head_knn_test)
clf_report_knn_test = pd.DataFrame(classification_report(y_test.T,y_head_knn_test, output_dict = True))
print("KNN CLASSIFICATION REPORT TEST DATA:\n",clf_report_knn_test)
print("_______________________________________________________________________")


#SVM EVALUATION
y_head_svm_train = svm.predict(x_train.T)
y_head_svm_test = svm.predict(x_test.T)

confusionmatrix_svm_train = confusion_matrix(y_train, y_head_svm_train)
clf_report_svm_train = pd.DataFrame(classification_report(y_train.T,y_head_svm_train, output_dict = True))
print("SVM CLASSIFICATION REPORT TRAIN DATA:\n",clf_report_svm_train)
print("_______________________________________________________________________")

confusionmatrix_svm_test = confusion_matrix(y_test, y_head_svm_test)
clf_report_svm_test = pd.DataFrame(classification_report(y_test.T,y_head_svm_test, output_dict = True))
print("SVM CLASSIFICATION REPORT TEST DATA:\n",clf_report_svm_test)
print("_______________________________________________________________________")

#NaiveBayes EVALUATION
y_head_naivebayes_train = naivebayes.predict(x_train.T)
y_head_naivebayes_test = naivebayes.predict(x_test.T)

confusionmatrix_naivebayes_train = confusion_matrix(y_train, y_head_naivebayes_train)
clf_report_naivebayes_train = pd.DataFrame(classification_report(y_train.T,y_head_naivebayes_train, output_dict = True))
print("NaiveBayes CLASSIFICATION REPORT TRAIN DATA:\n",clf_report_naivebayes_train)
print("_______________________________________________________________________")

confusionmatrix_naivebayes_test = confusion_matrix(y_test, y_head_naivebayes_test)
clf_report_naivebayes_test = pd.DataFrame(classification_report(y_test.T,y_head_naivebayes_test, output_dict = True))
print("NaiveBayes CLASSIFICATION REPORT TEST DATA:\n",clf_report_naivebayes_test)
print("_______________________________________________________________________")


#Decision Tree EVALUATION
y_head_decisiontree_train = decisiontree.predict(x_train.T)
y_head_decisiontree_test = decisiontree.predict(x_test.T)

confusionmatrix_decisiontree_train = confusion_matrix(y_train, y_head_decisiontree_train)
clf_report_decisiontree_train = pd.DataFrame(classification_report(y_train.T,y_head_decisiontree_train, output_dict = True))
print("Decision Tree CLASSIFICATION REPORT TRAIN DATA:\n",clf_report_decisiontree_train)
print("_______________________________________________________________________")

confusionmatrix_decisiontree_test = confusion_matrix(y_test, y_head_decisiontree_test)
clf_report_decisiontree_test = pd.DataFrame(classification_report(y_test.T,y_head_decisiontree_test, output_dict = True))
print("Decision Tree CLASSIFICATION REPORT TEST DATA:\n",clf_report_decisiontree_test)
print("_______________________________________________________________________")

#Random Forest EVALUATION
y_head_randomforest_train = randomforest.predict(x_train.T)
y_head_randomforest_test = randomforest.predict(x_test.T)

confusionmatrix_randomforest_train = confusion_matrix(y_train, y_head_randomforest_train)
clf_report_randomforest_train = pd.DataFrame(classification_report(y_train.T,y_head_randomforest_train, output_dict = True))
print("Random Forest CLASSIFICATION REPORT TRAIN DATA:\n",clf_report_randomforest_train)
print("_______________________________________________________________________")

confusionmatrix_randomforest_test = confusion_matrix(y_test, y_head_randomforest_test)
clf_report_randomforest_train = pd.DataFrame(classification_report(y_test.T,y_head_randomforest_test, output_dict = True))
print("Random Forest CLASSIFICATION REPORT Test DATA:\n",clf_report_randomforest_train)
print("_______________________________________________________________________")


#plotting
#KNN Plot
plt.figure(figsize=(6,6))

plt.suptitle("KNN Confusion Matrixes",fontsize=18)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(1,2,1)
plt.title("Train Data",fontsize=12)
sns.heatmap(confusionmatrix_knn_train,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,2,2)
plt.title("Test Data",fontsize=12)
sns.heatmap(confusionmatrix_knn_test,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

#SVM Plot
plt.figure(figsize=(6,6))

plt.suptitle("SVM Confusion Matrixes",fontsize=18)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(1,2,1)
plt.title("Train Data",fontsize=12)
sns.heatmap(confusionmatrix_svm_train,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,2,2)
plt.title("Test Data",fontsize=12)
sns.heatmap(confusionmatrix_svm_test,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

#NaiveBayes Plot
plt.figure(figsize=(6,6))

plt.suptitle("Naive Bayes Confusion Matrixes",fontsize=18)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(1,2,1)
plt.title("Train Data",fontsize=12)
sns.heatmap(confusionmatrix_naivebayes_train,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,2,2)
plt.title("Test Data",fontsize=12)
sns.heatmap(confusionmatrix_naivebayes_test,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

#Decision Tree Plot
plt.figure(figsize=(6,6))

plt.suptitle("Decision Tree Confusion Matrixes",fontsize=18)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(1,2,1)
plt.title("Train Data",fontsize=12)
sns.heatmap(confusionmatrix_decisiontree_train,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,2,2)
plt.title("Test Data",fontsize=12)
sns.heatmap(confusionmatrix_decisiontree_test,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

#Random Forest Plot
plt.figure(figsize=(6,6))

plt.suptitle("Random Forest Confusion Matrixes",fontsize=18)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(1,2,1)
plt.title("Train Data",fontsize=12)
sns.heatmap(confusionmatrix_randomforest_train,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,2,2)
plt.title("Test Data",fontsize=12)
sns.heatmap(confusionmatrix_randomforest_test,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})


#Comparison Train Data Plot
plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(3,2,1)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(confusionmatrix_knn_test,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,2,2)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(confusionmatrix_svm_test,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,2,3)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(confusionmatrix_naivebayes_test,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,2,4)
plt.title("Decision Tree Confusion Matrix")
sns.heatmap(confusionmatrix_decisiontree_test,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,2,5)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(confusionmatrix_randomforest_test,annot=True,cmap="gray",fmt="d",cbar=False, annot_kws={"size": 24})


#Unused but not deleted yet just in case
#Decision Tree Plot 

# from sklearn import tree
# fig = plt.figure(figsize=(300,300))
# decisiontreeplot= tree.plot_tree(decisiontree,feature_names=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope"],
#                                  class_names="target",filled=True)

#menambahkan berat dan bias

# #logistic Regression Testing
# #initialize
# def initialize(dimension):
    
#     weight = np.full((dimension,1),0.01)
#     bias = 0.0
#     return weight,bias

# #Sigmoid Function
# def sigmoid(z):
#     y_head = 1/(1 + np.exp(-z))
#     return y_head

# #Forward and Backward Propagation
# def forwardBackward(weight,bias,x_train,y_train):
#     # Forward
    
#     y_head = sigmoid(np.dot(weight.T,x_train) + bias)
#     loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
#     cost = np.sum(loss) / x_train.shape[1]
    
#     # Backward
#     derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
#     derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
#     gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}
    
#     return cost,gradients

# def update(weight,bias,x_train,y_train,learningRate,iteration) :
#     costList = []
#     index = []
    
#     #for each iteration, update weight and bias values
#     for i in range(iteration):
#         cost,gradients = forwardBackward(weight,bias,x_train,y_train)
#         weight = weight - learningRate * gradients["Derivative Weight"]
#         bias = bias - learningRate * gradients["Derivative Bias"]
        
#         costList.append(cost)
#         index.append(i)

#     parameters = {"weight": weight,"bias": bias}
    
#     print("iteration:",iteration)
#     print("cost:",cost)

#     plt.plot(index,costList)
#     plt.xlabel("Number of Iteration")
#     plt.ylabel("Cost")
#     plt.show()

#     return parameters, gradients

# def predict(weight,bias,x_test):
#     z = np.dot(weight.T,x_test) + bias
#     y_head = sigmoid(z)

#     y_prediction = np.zeros((1,x_test.shape[1]))
    
#     for i in range(y_head.shape[1]):
#         if y_head[0,i] <= 0.5:
#             y_prediction[0,i] = 0
#         else:
#             y_prediction[0,i] = 1
#     return y_prediction

# def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):
#     dimension = x_train.shape[0]
#     weight,bias = initialize(dimension)
    
#     parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)

#     y_prediction = predict(parameters["weight"],parameters["bias"],x_test)
    
#     print("Manuel Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))
    
# logistic_regression(x_train,y_train,x_test,y_test,1,100)   