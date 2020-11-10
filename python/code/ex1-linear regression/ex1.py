import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def cost_linear(X,y,theta):
	cost=X@theta.T
	temp=np.power(cost-y,2)
	return np.sum(temp)/(2*X.shape[0])
def gradientDescent(X,y,theta,alpha,num):
	temp=np.matrix(np.zeros(theta.shape[1]))
	cost=np.zeros(num)
	for i in range(num):
		error=X@theta.T-y
		for j in range(theta.shape[1]):
			term=np.multiply(error,X[:,j])
			temp[0,j]=theta[0,j]-alpha*np.sum(term)/X.shape[0]
		theta=temp
		cost[i]=cost_linear(X, y, theta)
	return theta,cost
		
	
data=pd.read_csv('ex1data1.txt',header=None,names=['population','profit'])
data.insert(0, 'Ones', 1)
col=data.shape[1]
X=data.iloc[ : , 0:col-1];y=data.iloc[ : ,col-1:col]#与y=data.iloc[ : ,col-1]做区别
theta=np.matrix([0,0]);
X=np.matrix(X);y=np.matrix(y)#矩阵化
print(cost_linear(X, y, theta))
num=1000
alpha=0.01
g,cost=gradientDescent(X, y, theta, alpha, num)
x = np.linspace(data.population.min(), data.population.max(), 100)
f=g[0,0]+g[0,1]*x
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.population,data.profit,label='Training Data')
ax.legend()
ax.set_xlabel('Population')
ax.set_ylabel('Profit')

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(num),cost,'r',label='Cost')
ax.legend()
ax.set_xlabel('Cost')
ax.set_ylabel('times')


plt.show()
