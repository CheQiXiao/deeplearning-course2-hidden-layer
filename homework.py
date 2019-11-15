import numpy
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
np.random.seed(1) #设置一个固定的随机种子，以保证接下来的步骤中我们的结果是一致的。
X,Y=load_planar_dataset()
#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral) #绘制散点图

# 上一语句如出现问题，请使用下面的语句：
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral) #绘制散点图
# shape_X=X.shape
# shape_Y=Y.shape
# m=Y.shape[1]
# print("X的维度为：" + str(shape_X))
# print("Y的维度为："+str(shape_Y))
# print("共有数据"+str(m)+"个")
#利用已有的工具进行拟合
# clf=sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T,np.squeeze(Y.T))
# plot_decision_boundary(lambda x: clf.predict(x), X, Y) #绘制决策边界
# plt.title("Logistic Regression") #图标题
# LR_predictions  = clf.predict(X.T) #预测结果
# print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
# 		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
#        "% " + "(正确标记的数据点所占的百分比)")
def layer_sizes(X,Y):

    """
    参数：
    X - 输入数据集,维度为（输入的数量，训练/测试的数量）
    Y - 标签，维度为（输出的数量，训练/测试数量）

    返回：
        n_x - 输入层的节点数量
        n_h - 隐藏层的节点数量
        n_y - 输出层的节点数量
        """
    n_x=X.shape[0]
    n_h=4
    n_y=Y.shape[0]
    return (n_x,n_h,n_y)

# X_asses,Y_asses=layer_sizes_test_case()
# (n_x,n_h,n_y)=layer_sizes(X_asses,Y_asses)
#测试layer_sizes???????为什么要设置为(5,3)  (2,3),!!!!这里是测试用的，layer_sizes_test_case()测试随机神经网络的结构
# print("=========================测试layer_sizes=========================")
# X_asses , Y_asses = layer_sizes_test_case()
# (n_x,n_h,n_y) =  layer_sizes(X_asses,Y_asses)
# print("输入层的节点数量为: n_x = " + str(n_x))
# print(X_asses)
# print("隐藏层的节点数量为: n_h = " + str(n_h))
# print("输出层的节点数量为: n_y = " + str(n_y))
# print(Y_asses)
def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    W1=np.random.rand(n_h,n_x)*0.01
    b1=np.zeros(shape=(n_h,1))
    W2=np.random.rand(n_y,n_h)*0.01
    b2=np.zeros(shape=(n_y,1))

    assert (W1.shape == ( n_h , n_x ))
    assert (b1.shape == ( n_h , 1 ))
    assert (W2.shape == ( n_y , n_h ))
    assert (b2.shape == ( n_y, 1))

    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2

    }
    return parameters

#测试initialize_parameters
# print("=========================测试initialize_parameters=========================")
# n_x,n_h,n_y=initialize_parameters_test_case()
# parameters=initialize_parameters(n_x,n_h,n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
def forward_propagation(X,parameters):
    """

    :param X:维度为（n_x，m）的输入数据。
    :param parameters:初始化函数（initialize_parameters）的输出
    :return:
        A2 - 使用sigmoid()函数计算的第二次激活后的数值
         cache - 包含“Z1”，“A1”，“Z2”和“A2”的字典类型变量
    """
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    Z1=np.dot(W1,X)+b1
    A1=sigmoid(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)

    assert (A2.shape == (1,X.shape[1]))#???????X.shape[1]=400  A2.shape应该和Y.shape是一样的 都是（1,400）
    cache={
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2,

    }
    return (A2,cache)
# #测试forward_propagation
# print("=========================测试forward_propagation=========================")
# X_asses,parameters=forward_propagation_test_case()
# A2,cache=forward_propagation(X_asses,parameters)
# print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))
#计算损失值
def compute_cost(A2,Y,parameters):
    """

    :param A2:使用sigmoid()函数计算的第二次激活后的数值
    :param Y:True"标签向量,维度为（1，数量）
    :param parameters:一个包含W1，B1，W2和B2的字典类型的变量
    :return:成本 - 交叉熵成本给出方程
    """
    m=Y.shape[1]
    W1 = parameters["W1"]#????需要用到W1 W2吗

    W2 = parameters["W2"]
    #logprobs=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),1-Y)
    logprobs=np.dot(np.log(A2+1e-5),Y.T)+np.dot(np.log(1-A2+1e-5),(1-Y).T)
    cost = - 1.0 / m * logprobs[0][0];
    cost=float(np.squeeze(cost))
    assert (isinstance(cost,float))
    return cost

#测试compute_cost
# print("=========================测试compute_cost=========================")
# A2,Y_access,parameters=compute_cost_test_case()
# cost=compute_cost(A2,Y_access,parameters)
# print("cost="+str(cost))
def backward_propagation(parameters,cache,X,Y):
    """

    :param parameters:包含我们的参数的一个字典类型的变量
    :param cache:包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
    :param X:输入数据，维度为（2，数量）
    :param Y:“True”标签，维度为（1，数量）
    :return:grads - 包含W和b的导数一个字典类型的变量。
    """
    m=Y.shape[1]
    W1=parameters["W1"]
    W2=parameters["W2"]
    A1=cache["A1"]
    A2=cache["A2"]

    dZ2=A2-Y
    dW2=(1/m)*np.dot(dZ2,A1.T)
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1=np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))
    dW1=(1/m)*np.dot(dZ1,X.T)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)

    grads={
        "dW1":dW1,
        "dW2":dW2,
        "db1":db1,
        "db2":db2
    }
    return grads
# print("=========================测试backward_propagation=========================")
# parameters,cache,X_assess,Y_assess=backward_propagation_test_case()
# grads=backward_propagation(parameters,cache,X_assess,Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))
def update_parameters(paramaters,grads,learning_rate=1.2):
    W1,W2=paramaters["W1"],paramaters["W2"]
    b1,b2=paramaters["b1"],paramaters["b2"]

    dW1,dW2=grads["dW1"],grads["dW2"]
    db1,db2=grads["db1"],grads["db2"]

    W1=W1-learning_rate*dW1
    W2=W2-learning_rate*dW2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2

    paramaters={
        "W1":W1,
        "W2":W2,
        "b1":b1,
        "b2":b2
    }
    return paramaters
#测试update_parameters
# print("=========================测试update_parameters=========================")
# paramaters,grads=update_parameters_test_case()
# paramaters=update_parameters(paramaters,grads)
# print("W1="+str(paramaters["W1"]))
# print("W2="+str(paramaters["W2"]))
# print("b1="+str(paramaters["b1"]))
# print("b2="+str(paramaters["b2"]))

def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    """

    :param X:
    :param Y:
    :param n_h:隐藏层节点的数量
    :param num_iterations: 梯度下降循环迭代的次数
    :param print_cost:如果为True，则每1000次迭代打印一次成本数值
    :return:
    """
    np.random.seed(3)
    n_x=layer_sizes(X,Y)[0]
    n_y=layer_sizes(X,Y)[2]


    parameters=initialize_parameters(n_x,n_h,n_y)
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2 = parameters["W2"]
    b2=parameters["b2"]

    for i in range(num_iterations):

        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y,parameters)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=0.05)#????参数更新出现了问题

        if print_cost:
            if i%1000==0:
                print("第 ",i,"次循环，成本是"+str(cost))
    return parameters
#测试nn_model
# print("=========================测试nn_model=========================")
# X_assess,Y_assess=nn_model_test_case()
# paramaters = nn_model(X_assess,Y_assess,4,num_iterations=10000,print_cost=True)
# print("W1 = " + str(paramaters["W1"]))
# print("b1 = " + str(paramaters["b1"]))
# print("W2 = " + str(paramaters["W2"]))
# print("b2 = " + str(paramaters["b2"]))

def predict(parameters,X):
    """

    :param parameters:
    :param X:
    :return: predictions - 我们模型预测的向量（红色：0 /蓝色：1）
    """
    A2,cache=forward_propagation(X,parameters)
    predictions=np.round(A2)
    return predictions

#测试predict??????输出总是1
print("=========================测试predict=========================")
parameters, X_assess=predict_test_case()
predictions = predict(parameters,X_assess)
print("prediction="+ str(np.mean(predictions)))
parameters = nn_model(X, Y, n_h = 25, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x:predict(parameters,x.T),X,Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
predictions=predict(parameters,X)
print('准确率：%d' % float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/float(Y.size)*100)+'%')
plt.show()