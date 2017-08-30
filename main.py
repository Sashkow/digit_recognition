import pandas as pd 
import numpy as np 

# def sigmoida(x):
# 	"""
# 	input value can be number or np.array
# 	"""

# 	return 1/(1+np.exp(-x))

# # a = np.array([[2,5,6],[5,6,9]])
# # print(sigmoida(a))

# train_data = pd.read_csv('train.csv')
# y = np.array(train_data.label)
# x = np.array(train_data.drop('label',axis=1))



# w1 = np.random.random((784,28))
# w2 = np.random.ramdom((28,10))

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in range(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

print(np.dot(np.dot([1,0,1],syn0),syn1))

