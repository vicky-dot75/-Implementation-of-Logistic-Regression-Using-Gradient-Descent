# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages

2. Read the dataset.

3. Define X and Y array.

4. Define a function for costFunction,cost and gradient


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: vignesh s
RegisterNumber:  25014344
*/
import pandas as pd
import numpy as np

data=pd.read_csv("/content/Placement_Data (1).csv")

data.head()
data1=data.copy()
data1.head()
data1=data.drop(['sl_no','salary'],axis=1)
data1

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])

X=data1.iloc[:,: -1]
Y=data1["status"]
theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+ (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)

  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5 , 1,0)
  return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)

print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)

```

## Output:
<img width="1376" height="526" alt="Screenshot 2025-12-24 170202" src="https://github.com/user-attachments/assets/c32cc6b6-118c-487c-8d64-a85b2700fa56" />

<img width="1349" height="530" alt="Screenshot 2025-12-24 170208" src="https://github.com/user-attachments/assets/1d91bca0-d409-455f-96d3-7a5c1739f5f0" />

<img width="1387" height="697" alt="Screenshot 2025-12-24 170218" src="https://github.com/user-attachments/assets/4d018640-3ea6-46b4-aee1-d1181a421ec3" />

<img width="1357" height="373" alt="Screenshot 2025-12-24 170225" src="https://github.com/user-attachments/assets/7ff160f1-29f7-48f1-b665-cbc137d5ab8d" />

<img width="1339" height="249" alt="Screenshot 2025-12-24 170232" src="https://github.com/user-attachments/assets/a88cfe0a-e112-490d-8ddb-72bca5ac7d15" />

<img width="673" height="123" alt="Screenshot 2025-12-24 170237" src="https://github.com/user-attachments/assets/e1eac904-6691-4267-8524-de316083e6a2" />

<img width="643" height="135" alt="Screenshot 2025-12-24 170244" src="https://github.com/user-attachments/assets/82dda739-2b45-4edb-a80b-4912b69d23a7" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

