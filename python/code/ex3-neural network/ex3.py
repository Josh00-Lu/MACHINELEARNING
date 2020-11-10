import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
def sigmoid(z):
	#sigmoid函数
	return 1/(1+np.exp(-z))
def cost(theta,X,y,learningRate):
	theta=np.matrix(theta)
	X=np.matrix(X)
	y=np.matrix(y)
	first=np.multiply(y,np.log(sigmoid(X*theta.T)))
	second=np.multiply(1-y,np.log(1-sigmoid(X*theta.T)))
	reg=learningRate/(2*X.shape[0])*np.sum(np.power(theta[:,1:theta.shape[1]],2))
	return np.sum(-first-second)/X.shape[0]+reg
def gradient(theta, X, y, learningRate):
	theta=np.matrix(theta)
	X=np.matrix(X)
	y=np.matrix(y)
	error=sigmoid(X*theta.T)-y
	grad=np.zeros(theta.shape[1])
	for i in range(X.shape[1]):
		if (i==0):
			grad[i]=np.sum(np.multiply(error,X[:,i]))/X.shape[0]
		else:
			grad[i]=(np.sum(np.multiply(error,X[:,i]))+learningRate*theta[:,i])/X.shape[0]
	return grad
def one_vs_all(X,y,num_labels,learning_rate):
	#num_labels为有多少类
	rows=X.shape[0]#X有多少行
	params=X.shape[1]
	all_theta=np.zeros((num_labels,params+1))
	X=np.insert(X,0,np.ones(rows),axis=1)
	for i in range(1,num_labels+1):
		theta=np.zeros(params+1)
		yi=np.array([1 if label==i else 0 for label in y])
		yi=yi.reshape((-1,1))
		fmin=minimize(fun=cost, x0=theta, jac=gradient,method='TNC', args=(X, yi,learning_rate))
		all_theta[i-1,:]=fmin['x']
	return all_theta
def predict_all(X,all_theta):
	rows=X.shape[0]
	params=X.shape[1]
	X=np.insert(X,0,np.ones(rows),axis=1)
	X=np.matrix(X)
	all_theta=np.matrix(all_theta)
	h=sigmoid(X*all_theta.T)
	h_ans=np.argmax(h,axis=1)+1
	return h_ans
	
data = loadmat('ex3data1.mat')
all_theta=one_vs_all(data['X'], data['y'], 10, 1)
hypo=predict_all(data['X'],all_theta)
correct=[1 if a==b else 0 for (a,b) in zip(hypo,data['y'])]
acurracy=sum(correct)/len(correct);
print('acurracy={}%'.format(acurracy*100));