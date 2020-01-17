
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#READING THE GIVEN DATA FILE 
dataFrame = pd.read_csv("ex1data2.txt", header=None)


#ASSIGNING X AND y values to train our model.
X = dataFrame[[0,1]].values
y = dataFrame[2].values.reshape((-1,1))


X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
y = (y - np.mean(y)) / np.std(y)

#plt.scatter(X,y, c="red", label="training data")

# ADDING X(o) to the training data

X = np.hstack((np.ones((X.shape[0],1)), X))


# Setting both weight to 0 initially.
w = np.zeros(X[0].size)


errors = []
#learning rate
alpha = 0.1
m = X.shape[0]
for _ in range(300):

    j = np.zeros(X[0].size).astype('float64')
    for i in range(m):
        
         for k in range(w.size):
             j[k]  += (X[i] @ w - y[i].item()) * X[i][k]
             
         
    errors.append((X[i] @ w - y[i].item()) ** 2)
    
    for k in range(w.size):
        w[k] = w[k] - alpha * ( 1/ m ) * j[k]
        
    print(f"{w[0]} and {w[1]}")   
       



plt.plot(X[ :,1] , X @ w, label="regression line" )


plt.legend()
plt.show()

plt.title("ERRORS ")
plt.plot(errors)
plt.xlabel("ITERATION COUNT")
plt.ylabel("SQUARED ERROR")