# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries

2. Set variables for assigning dataset values

3. Import linear regression from sklearn.

4. Assign the points for representing in the graph

5. Predict the regression for marks by using the representation of the graph. 

6. Compare the graphs and hence we obtained the linear regression for the given datas.
   


## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: YUVARAJ B
RegisterNumber:  212222230182

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='purple')
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 

```

## Output:
Data set:

![265497870-65b41726-0120-4a4b-b40a-d43e248823d9](https://github.com/Yuva2005raj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343998/56c2da16-b6d2-4c26-ae7b-625c5bd3da57)

Head value:

![265497944-240b2352-15b4-43d4-a1e9-5bf7cff9c73c](https://github.com/Yuva2005raj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343998/3ca8f16a-09cd-4801-9267-4619f38aa0f1)

Tail value:

![265498034-74f38c0c-076a-4468-b7ec-beda04215aa7](https://github.com/Yuva2005raj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343998/6e8373d0-c2df-4d90-bd6e-90ce8cf643b4)

X and Y value:

![265498293-93dc266d-8193-46b9-aacb-f235729cd7eb](https://github.com/Yuva2005raj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343998/4238496a-8fcf-4cf9-8884-c21c405e10f0)

Prediction values of x and y:

![265498417-1c6dbe39-5ea4-4fad-9f86-4115bcb5f5b6](https://github.com/Yuva2005raj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343998/1ea540bf-7f41-41f0-bf1f-23190154ed81)

MSE,MAE AND RMSE:

![265498528-7bbe9578-3bc9-4420-8f78-1c84e39fe21f](https://github.com/Yuva2005raj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343998/70b84265-d751-41eb-be93-d72184b8a5f2)

Trainning set:

![265499664-8719ffc5-5bc7-4a8e-9002-42f8be22da89](https://github.com/Yuva2005raj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343998/5947d1d9-6eb2-4d90-87ea-2ca49628e47a)

Testing set:

![265499732-2afee5df-91f6-4421-b922-c7fa554885fd](https://github.com/Yuva2005raj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343998/37d77fc3-8c99-4d33-a6d3-5aaec99dbc68)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
