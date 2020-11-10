import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def cost_logic(theta,X,y):
	theta=np.matrix(theta)
	X=np.matrix(X)
	y=np.matrix(y)
	h=sigmond(X*theta.T)
	first=np.multiply(y,np.log(h+1e-5))
	second=np.multiply((1-y),np.log(1-h+1e-5))
	return np.sum(-first-second)/X.shape[0]
def gradient(theta,X,y):
	theta=np.matrix(theta)
	X=np.matrix(X)
	y=np.matrix(y)
	grad=np.zeros(theta.shape[1])
	error=sigmond(X*theta.T)-y
	for j in range(theta.shape[1]):
		grad[j]=np.sum(np.multiply(error,X[:,j]))/X.shape[0]
	return grad
def showdata(data):
	fig,ax=plt.subplots(figsize=(12,8))
	group1=data[data['Admitted']==1]#group1=data[data.Admitted.isin([0])]
	group2=data[data['Admitted']==0]
	ax.scatter(group1['Exam1'],group1['Exam2'],c='b',marker='o',label='Admitted')
	ax.scatter(group2['Exam1'],group2['Exam2'],c='r',marker='x',label='Not Admitted')
	ax.legend()
	ax.set_xlabel('Exam 1 Score')
	ax.set_ylabel('Exam 2 Score')
	plt.show()
def sigmond(z):
	return 1/(1+np.exp(-z))
data=pd.read_csv('ex2data1.txt',header=None,names=['Exam1','Exam2','Admitted'])
#showdata(data)
data.insert(0,'Ones',1)
col=data.shape[1]
X=data.iloc[ : , 0:col-1];y=data.iloc[ : ,col-1:col]
theta=np.zeros(X.shape[1])
X=np.array(X);y=np.array(y);
#print(gradient(X, y,theta))
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost_logic, x0=theta, fprime=gradient, args=(X, y))
print(cost_logic(result[0],X,y))
theta=np.zeros(X.shape[1])
result = opt.minimize(fun=cost_logic, x0=theta, jac=gradient, args=(X, y))
print(cost_logic(result['x'],X,y))
'''
num=np.arange(-10,10)
fig,ax=plt.subplots()
ax.plot(num,sigmond(num))
plt.show()
'''

