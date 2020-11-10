import numpy as np
def cost_linear(X,y,theta):
	cost=X@theta.T
	temp=pow(cost-y,2)
	return np.sum(temp)/(2*X.shape[0])
def cost_logic(X,y,theta):
	X=np.matrix(X)
	y=np.matrix(y)
	theta=np.matrix(theta)
	cost=1/(1+np.exp(-X@theta.T))
	first=-y@np.log(cost)
	second=-(1-y)@np.log(1-cost)
	return np.sum(first+second)/X.shape[0]

def cost_linear_regulated(X,y,theta,rate):
	reg=pow(theta[1:],2)
	return cost_linear(X,y,theta)+rate/(2*X.shape[0])*sum(reg)

def cost_logic_regulated(X,y,theta,rate):
	reg=pow(theta[1:],2)
	return cost_logic(X,y,theta)+rate/(2*X.shape[0])*sum(reg)
	
X=np.array([
	[1,2,3,4],
	[5,6,7,8],
	[9,10,11,12]
])
theta=np.array([1,2,3,4])#theta是一个行向量
y1=np.array([3,4,5]).reshape(-1,1)
y2=[0,1,0]
print('线性回归结果',cost_linear(X,y1,theta))

print('逻辑回归结果',cost_logic(X, y2, theta))

print('线性回归正则化结果',cost_linear_regulated(X,y1,theta,1))

print('逻辑回归正则化结果',cost_logic_regulated(X,y2,theta,1))