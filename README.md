# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: G.HINDHU
RegisterNumber:  212223230079
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```


## Output:
### data
![image](https://github.com/user-attachments/assets/dd848f0b-4696-48d2-ad89-2d40ee146783)


### data.shape()
![image](https://github.com/user-attachments/assets/8d08c385-b99d-474a-97b2-b1b983897787)

### x.shape()
![image](https://github.com/user-attachments/assets/e9491612-1256-4246-83aa-a5220826b542)


### y.shape()  
![image](https://github.com/user-attachments/assets/bae9f1e0-65fa-402c-b8fb-69d4fc3ecc14)


### x_train
![image](https://github.com/user-attachments/assets/74143d16-ed5c-4207-8bde-7a0497d6bdc8)



### x_train.shape()
![image](https://github.com/user-attachments/assets/fab8cea0-6260-4964-b673-db58d08341f9)

### y_pred
![image](https://github.com/user-attachments/assets/4e1517fe-bd9d-4dae-a1ed-aaee601d6392)



### acc (accuracy)
![image](https://github.com/user-attachments/assets/6ece62d5-1d77-4197-903e-8c38ae87d0b4)

### con (confusion matrix)
![image](https://github.com/user-attachments/assets/1fb7b347-dd21-47df-b51e-13bc3263f9ba)


### cl (classification report)
![image](https://github.com/user-attachments/assets/3cacbd49-8e44-4109-9a86-e5c70a4e26fa)









## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
