import os
import pickle
from random import shuffle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import sys
sys.path.append('/Users/anithas/Desktop/ML_project')
from data_preparation import *


##########------------###############
## MAIN CODE ##
##########------------###############

# Reading the flist
flist = '/Users/anithas/Desktop/MICCAI_BraTS2020_TrainingData/Data_list.flist'
Subject_LIST = []
with open(flist, 'r') as fid:
    for line in fid:
        Subject_LIST.append(line.strip())

shuffle(Subject_LIST)
# Obtaining data
print ('----------------------')
print ('Obtaining data........')
print ('----------------------')
DATA, LABELS = prepare_data(subject_flist=Subject_LIST, cases_to_read=100)

print ('----------------------------')
print ("Data loaded Succesfully.....")
print ('----------------------------')

print ('-----------------------------------------------------------------------')
print ("Please wait while the data is being prepared for training & testing....")
print ('-----------------------------------------------------------------------')

print ('------------------------------------------------')
print ("Please wait while the data is being prepared....")
print ('------------------------------------------------')
FINAL_DATA = np.array(DATA)
print(FINAL_DATA.shape)
FINAL_LABEL = np.array(LABELS)

DATA = []
LABEL = []

'''Reshape 3D data into 2D'''
data_2D = []
for img in range(0, len(FINAL_DATA)):
    temp = FINAL_DATA[img, :, :]
    reshaped_temp = temp.reshape(1, 240 * 240)
    data_2D.append(np.concatenate(reshaped_temp, axis=0))
data_2D = np.array(data_2D)
output_2D = FINAL_LABEL

FINAL_DATA = []
FINAL_LABEL = []

'''Test-Train split'''
test_ratio = 0.2
shuffled_indices = np.random.permutation(len(data_2D))
test_set_size = int(len(data_2D) * test_ratio)
test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]
X_train, X_test = data_2D[train_indices], data_2D[test_indices]
Y_train, Y_test = output_2D[train_indices], output_2D[test_indices]

print ('------------------------------------')
print ("DONE.. Data is now ready to use....")
print ('------------------------------------')

print ('------------------------------------------------------------------')
print ("Creating a Simple RandomForestClassifier and starting training....")
print ('------------------------------------------------------------------')

''' RandomForestClassifier (Accuracy)'''
model = RandomForestClassifier(random_state=42, max_depth=23, n_estimators=100)
model.fit(X_train, Y_train)
filename = '/Users/anithas/Desktop/MICCAI_BraTS2020_TrainingData/RandomForestClassifier.sav'
pickle.dump(model, open(filename, 'wb'))

## Testing the model
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
Y_pred = loaded_model.predict(X_test)


'''Confusion matrix, Accuracy, sensitivity and specificity'''
cm1 = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix : \n', cm1)

total1 = sum(sum(cm1))
accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
print('Sensitivity : ', sensitivity1)

specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
print('Specificity : ', specificity1)



###########################################################################################################
#########################################...SGDClassifier...###############################################
###########################################################################################################

''' SGDClassifier (Normal)'''
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, Y_train)
filename = '/Users/anithas/Desktop/MICCAI_BraTS2020_TrainingData/SGD_Classifier.sav'
pickle.dump(sgd_clf, open(filename, 'wb'))

## Testing the model
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
Y_pred = loaded_model.predict(X_test)

cm1 = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix : \n', cm1)

total1 = sum(sum(cm1))
accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
print('Sensitivity : ', sensitivity1)

specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
print('Specificity : ', specificity1)

###########################################################################################################
#########################################...KNNClassifier (Minikowski)...##################################
# ###########################################################################################################

''' KNNClassifier (Minikowski)'''
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                               metric='minkowski')
y_predict_knn = cross_val_predict(knn_clf, X_train, Y_train, cv=3)

knn_clf.fit(X_train, Y_train)
filename = '/Users/anithas/Desktop/MICCAI_BraTS2020_TrainingData/KNN_Classifier.sav'
pickle.dump(knn_clf, open(filename, 'wb'))

## Testing the model
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
Y_pred = loaded_model.predict(X_test)

cm1 = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix : \n', cm1)

total1 = sum(sum(cm1))
accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
print('Sensitivity : ', sensitivity1)

specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
print('Specificity : ', specificity1)