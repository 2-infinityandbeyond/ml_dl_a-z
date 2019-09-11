import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X =dataset.iloc[:,0:-1].values
Y =dataset.iloc[:,-1].values
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
#imputer = imputer.fit(X[:,1:3])
#X[:,1:3]= imputer.transform(X[:,1:3])



# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=1/3,random_state=0)

#ftiing simple liner regression to training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#prediction
y_pred = regressor.predict(x_test)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.title('real values of test st to predicted values')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()