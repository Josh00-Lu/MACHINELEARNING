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
data=pd.read_csv('ex1data2.txt',header=None,names=['Size', 'Bedrooms', 'Price'])
data=(data-data.mean())/data.std()
data.insert(0,'Ones',1)
X=data.iloc[:,0:data.shape[1]-1]
y=data.iloc[:,data.shape[1]-1:data.shape[1]]
X=np.matrix(X)
y=np.matrix(y)
theta=np.zeros((1,X.shape[1]))
alpha=0.01;num=1000
g,cost=gradientDescent(X, y, theta, alpha, num)
x1=np.linspace(data.Size.min(),data.Size.max(),30)
x2=np.linspace(data.Bedrooms.min(),data.Bedrooms.max(),30)
x1,x2=np.meshgrid(x1,x2)
f=g[0,0]+g[0,1]*x1+g[0,2]*x2
fig=plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data.Size,data.Bedrooms,data.Price,label='Training Data')
ax.plot_surface(x1,x2,f,cmap='rainbow')
ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

fig,ax=plt.subplots()
ax.plot(np.arange(num),cost,'r',label='Cost')
ax.set_xlabel('Times')
ax.set_ylabel('Costs')
plt.show()