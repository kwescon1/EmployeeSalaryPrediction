import pandas as pd   #for reading CSVs into dataframes
import numpy as np # for dealing with specific array operations
import matplotlib.pyplot as plt # for visual view of the data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv("data/salaries.csv")

print(df.head())

x= df.iloc[:,:-1].values # get all rows with all columns except the last one
y = df.iloc[:,-1].values # get all rows with only the last column

# splitting data into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


model = LinearRegression()
model.fit(x_train,y_train)

# predict y_pred values after training our model
y_pred = model.predict(x_test)

print(y_pred)


# compare pred and actual values
plt.scatter(x_test,y_test)

plt.plot(x_test,y_pred,color="yellow")

plt.show()


# Evaluate Model Accuracy
r2 = r2_score(y_test,y_pred)
print(f"R2 Score: {r2} ({r2:.2%})")