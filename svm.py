############################## problem 1 ##########################
import pandas as pd 
import numpy as np 
import seaborn as sns

salary_train = pd.read_csv("C:/Users/usach/Desktop/Black box technquie-SVM/SalaryData_Train (1).csv")
salary_test = pd.read_csv("C:/Users/usach/Desktop/Black box technquie-SVM/SalaryData_Test (1).csv")
salary_train.head()
salary_train.describe()
salary_train.columns


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Salary_train = salary_train.apply(le.fit_transform)
Salary_test = salary_test.apply(le.fit_transform)

from sklearn.svm import SVC

salary_train_x = Salary_train.iloc[:,:13]
salary_train_y = Salary_train.iloc[:,13]
salary_test_x = Salary_test.iloc[:,:13]
salary_test_y = Salary_test.iloc[:,13]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(salary_train_x,salary_train_y)
pred_test_linear = model_linear.predict(salary_test_x)

np.mean(pred_test_linear==salary_test_y) # Accuracy = 80.41%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(salary_train_x,salary_train_y)
pred_test_poly = model_poly.predict(salary_test_x)

np.mean(pred_test_poly==salary_test_y) #accuracy =81.95%

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(salary_train_x,salary_train_y)
pred_test_rbf = model_rbf.predict(salary_test_x)

np.mean(pred_test_rbf==salary_test_y) #accuracy = 81.18%
# Constructing the confusion matrix f
from sklearn.metrics import confusion_matrix
confusion_matrix(salary_test_y,pred_test_linear)
confusion_matrix(salary_test_y,pred_test_poly)
confusion_matrix(salary_test_y,pred_test_rbf)

#ploy kernel is performing well 
####################### problem 2 ########################
import pandas as pd 
import numpy as np 
import seaborn as sns

forestfire = pd.read_csv("C:/Users/usach/Desktop/Black box technquie-SVM/forestfires.csv")

# removing columns which is not required
forest = forestfire.iloc[:,2:]

from sklearn import preprocessing
le= preprocessing.LabelEncoder()
forest['size_category']= le.fit_transform(forest['size_category'])

forest_in = forest.iloc[:,:28]
forest_out = forest.iloc[:,28]

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(forest_in,forest_out)

from sklearn.svm import SVC

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_x,train_y)
pred_test_linear = model_linear.predict(test_x)

np.mean(pred_test_linear==test_y) # Accuracy = 99.23%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_x,train_y)
pred_test_poly = model_poly.predict(test_x)
np.mean(pred_test_poly==test_y)# accuracy =80.76%

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_x,train_y)
pred_test_rbf = model_rbf.predict(test_x)
 
np.mean(pred_test_rbf==test_y) # 78.46%
#linear kernel  is performing well