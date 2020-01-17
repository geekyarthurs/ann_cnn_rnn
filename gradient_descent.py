
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#READING THE GIVEN DATA FILE 
dataFrame = pd.read_csv("ex1data1.txt", header=None)


#ASSIGNING X AND y values to train our model.
X = dataFrame[0].values.reshape(-1,1)
y = dataFrame[1].values.reshape(-1,1)
y = (y - np.mean(y)) / np.std(y)
X= (X - np.mean(X)) / np.std(X)

plt.scatter(X,y, c="red", label="training data")

# ADDING X(o) to the training data

X = np.hstack((np.ones((X.size,1)), X))


# Setting both weight to 0 initially.
w = np.zeros(X[0].size)

J1 = J2 = 0
errors = []
#learning rate
alpha = 0.01

m = X.shape[0]
for _ in range(10000):

    J1 = J2 = 0
    for i in range(m):
         J1  += (X[i] @ w - y[i].item()) * X[i][0]
         J2  += (X[i] @ w - y[i].item()) * X[i][1] 
         
    errors.append((X[i] @ w - y[i].item()) ** 2)
    w[0] = w[0] - alpha * ( 1/ m ) * J1
    w[1] = w[1] - alpha * ( 1/ m ) * J2
    print(f"{w[0]} and {w[1]}")   
       



plt.plot(X[ :,1] , X @ w, label="regression line" )


plt.legend()
plt.show()

plt.title("ERRORS ")
plt.plot(errors)
plt.xlabel("ITERATION COUNT")
plt.ylabel("SQUARED ERROR")