# Multivariant linear regression example

# import necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("data/salaries-cities.csv")


print(df.head())

x = df.iloc[:,:-1] # Take all columns except the last colum
y = df.iloc[:,-1] # Take all rows of the last column


#split data into 80 percent train data and 20 percent test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#define the model to use
model = LinearRegression()

#train the model with you train data
model.fit(x_train,y_train)

#predict the target values for the x_test values
y_pred = model.predict(x_test)

print(y_pred)


#check accuracy of prediction
# check the actual y_test values against the predicted values
r2 = r2_score(y_test,y_pred)

print(f"R2 score: {r2} {r2:.2%}")



#predict salaries based on year and city
salaries = model.predict([[11,1], [11,2],[12,1],[12,2]])

print(salaries)



