import pandas as pd   #for reading CSVs into dataframes
import numpy as np # for dealing with specific array operations
import matplotlib.pyplot as plt # for visual view of the data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("data/salaries.csv")

# print(df.head())


# split dataframe into feature(x) and target(y) values
# we use a dataframe property called iloc[] from pandas to do the splitting

x= df.iloc[:,:-1].values # get all rows with all columns except the last one
y = df.iloc[:,-1].values # get all rows with only the last column


 # specific row and column
 # print(df.iloc[1,2]) this will output second row and its third column

 # ALWAYS REMEMBER, COUNTING STARTS FROM 0

 # print(df.iloc[-1,0]) print last row first column

 # RANGE OF ROWS / COLUMNS
# print(df.iloc[1:3,0]) Syntax with a colon: rows from number 1 to number 3(not included), with their first column.

# print(df.iloc[:,0]) Colon without numbers: get ALL rows, with their first column (colon without numbers acts as "all").


# print(x[0:5])
# print(y[0:5])
# All rows/columns with multiple columns/rows.
#
# print(df.iloc[:, 0:2])  Colons for rows/column: get ALL rows, with their first two columns.


# The next step is to load the data as a chart, showing the salary dependency on the years of experience. We expect it to be linear, which means  - the more years the person works, the more proportionally they earn
# plt.scatter(x,y)

# plt.show()

# splitting data into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# print(x_train.shape)
# print(x_test.shape)


model = LinearRegression()
model.fit(x_train,y_train)

# predict y_pred values after training our model
y_pred = model.predict(x_test)

print(y_pred)


# compare pred and actual values
plt.scatter(x_test,y_test)

plt.plot(x_test,y_pred,color="yellow")

# plt.scatter(x,y)
plt.show()


# Evalute Model Accuracy
