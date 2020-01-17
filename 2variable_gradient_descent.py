
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

J1 = J2 = J3 = 0
errors = []
#learning rate
alpha = 0.1

m = X.shape[0]
for _ in range(100):

    J1 = J2 = J3 =  0
    for i in range(m):
        
         J1  += (X[i] @ w - y[i].item()) * X[i][0]
         J2  += (X[i] @ w - y[i].item()) * X[i][1] 
         J3  += (X[i] @ w - y[i].item()) * X[i][2]
         
   
    w[0] = w[0] - alpha * ( 1/ m ) * J1
    w[1] = w[1] - alpha * ( 1/ m ) * J2
    w[2] = w[2] - alpha * ( 1/ m ) * J3
    print(f"{w[0]} and {w[1]}")   
       



plt.plot(X[ :,1] , X @ w, label="regression line" )


plt.legend()
plt.show()

plt.title("ERRORS ")
plt.plot(errors)
plt.xlabel("ITERATION COUNT")
plt.ylabel("SQUARED ERROR")