import numpy as np
import matplotlib.pyplot as plt
def generate_data():

    np.random.seed(42)  
    x = np.random.rand(100, 10)  
    w = np.random.rand(10, 1) 
    b=np.random.rand(1)      
    y = x @ w +b                  
    return x, y
def generate_data_with_noise():
    np.random.seed(42)  
    x = np.random.rand(100, 10)  
    w = np.random.rand(10, 1)   
    b=np.random.rand(1)  
    y = x @ w + b+np.random.normal(0, 0.1, (100, 1))  
    return x, y
class LinearRegression:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.w=None
        self.b=None
    def slove(self,x,y):
        x=np.c_[self.x,np.ones((x.shape[0], 1))] 
        W=np.linalg.inv(x.T@x) @ x.T @ y
        self.w=W[:-1]
        self.b=W[-1]
        return self.w, self.b
    
    def predict(self, x):
        return x @ self.w + self.b
    def r2_score(self, y_test, y_pred):
        res = np.sum((y_test - y_pred) ** 2)
        tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (res / tot)
#有噪声数据
x,y=generate_data()
x_train=x[:80]
y_train=y[:80]
x_test=x[80:]
y_test=y[80:]
linear=LinearRegression(x_train,y_train)
w, b = linear.slove(x_train, y_train)
y_pred = linear.predict(x_test)
r2 = linear.r2_score(y_test, y_pred)
print(f"w: {w.flatten()}, b: {b.flatten()}, R^2: {r2:.4f}")

#无噪声数据
x1,y1=generate_data_with_noise()
x_train1=x1[:80]
y_train1=y1[:80]
x_test1=x1[80:]
y_test1=y1[80:]
linear1=LinearRegression(x_train1,y_train1)
w1, b1 = linear1.slove(x_train1, y_train1)
y_pred1 = linear1.predict(x_test1)
r2_1 = linear1.r2_score(y_test1, y_pred1)
print(f"w: {w1.flatten()}, b: {b1.flatten()}, R^2: {r2_1:.4f}")

#绘制图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'k--', lw=2)  
plt.xlabel('true')
plt.ylabel('pred')
plt.title(f'true vs pred (R²={r2:.4f})')
plt.grid(True)


plt.subplot(1, 2, 2)
plt.scatter(y_test1, y_pred1, color='red')
plt.plot([y_test1.min(), y_test1.max()], [y_test1.min(), y_test1.max()], 
         'k--', lw=2)
plt.xlabel('true')
plt.ylabel('pred')
plt.title(f'true vs pred (R²={r2_1:.4f})')
plt.grid(True)
plt.tight_layout()
plt.show()


  