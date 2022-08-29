# 神经网络类的定义
# neural Network class definition
import numpy as np
import scipy.special as spe  #用于设置激活函数 / using to set the activation function
import pickle                #用于读取权重文件 / using to read the weights file
import matplotlib.pyplot     #用于显示反向输出的图像 / using for show the image of BackQuery

class NeuralNetwork:
    #构造函数
    #Constructor
    def __init__(self,inputnodes,hiddennodes_1,hiddennodes_2,outputnodes,learning_rate,w_ih,w_hh,w_ho):
        # 设置节点数 / set nodes numbers
        self.inodes = inputnodes
        self.hnodes_1 = hiddennodes_1
        self.hnodes_2 = hiddennodes_2
        self.onodes = outputnodes
        # 设置学习率 / set learning rate
        self.lr  = learning_rate
        # 初始化权重 / set weights
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.w_ho = w_ho
        # 激活函数 / set the activation function
        self.active_fun = lambda x: spe.expit(x)
        self.inverse_activation_function = lambda x: spe.logit(x)
        pass

    #查询函数。传入输入变量，输出识别结果
    #Query functions. input variables and outputs are recognition results
    def query(self,inputs_list):
        # 把输入转换成二维矩阵 / Converts the input to a two-dimensional matrix
        inputs = np.array(inputs_list,ndmin=2).T
        # 正向传播计算 / Forward propagation calculations
        hidden_1_inputs = np.dot(self.w_ih, inputs)
        hidden_1_outputs = self.active_fun(hidden_1_inputs)
        hidden_2_inputs = np.dot(self.w_hh, hidden_1_outputs)
        hidden_2_outputs = self.active_fun(hidden_2_inputs)
        out_inputs = np.dot(self.w_ho, hidden_2_outputs)
        out_outputs = self.active_fun(out_inputs)
        return out_outputs

    #反向查询函数。输入是要查询的数字，输出是这个数字根据权重生成的图像。用于查看网络学到了什么
    #BackQuery function. The input is the number to be queried,
    #and the output is the image that this number generates based on its weights.
    #Used to see what the web has learned
    def backquery(self, targets_list):
        # 将目标列表转置为垂直阵列
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T

        # 计算进入最终输出层的信号
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # 计算出隐藏层的信号
        # calculate the signal out of the hidden layer
        hidden_2_outputs = np.dot(self.w_ho.T, final_inputs)
        # 将它们缩放回 0.01 到 0.99
        # scale them back to 0.01 to .99
        hidden_2_outputs -= np.min(hidden_2_outputs)
        hidden_2_outputs /= np.max(hidden_2_outputs)
        hidden_2_outputs *= 0.98
        hidden_2_outputs += 0.01

        # 计算进入最终输出层的信号
        # calculate the signal into the final output layer
        hidden_2_inputs = self.inverse_activation_function(hidden_2_outputs)
        # 计算出隐藏层2的信号
        # calculate the signal out of the hidden2 layer
        hidden_1_outputs = np.dot(self.w_hh.T, hidden_2_inputs)

        hidden_1_outputs -= np.min(hidden_1_outputs)
        hidden_1_outputs /= np.max(hidden_1_outputs)
        hidden_1_outputs *= 0.98
        hidden_1_outputs += 0.01
        # 计算出隐藏层1的信号
        # calculate the signal into the hidden1 layer
        hidden_1_inputs = self.inverse_activation_function(hidden_1_outputs)
        # 计算出输入层的信号
        # calculate the signal out of the input layer
        inputs = np.dot(self.w_ih.T, hidden_1_inputs)

        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs


#----------------------------------------main-------------------------------------------------
# 以下的函数用于在main文件中调用
# The following functions are used to call in a main file

# 读取权重文件
# read weights file
def load_w(name):
    f = open(name+'.pkl', 'rb')
    # 使用load的方法将数据从pkl文件中读取出来
    w = pickle.load(f)
    # 关闭文件
    f.close()
    return w

# 初始化神经网络
# initialize a nerwork
def InitializeANN():
    # number of inputnodes,hiddennodes_1,hiddennodes_2outputnodes
    input_nodes = 784
    hidden_nodes_1 = 80
    hidden_nodes_2 = 40
    output_nodes = 10
    # learning rate
    learning_rate = 0.1
    # 读取各层的权重矩阵
    # load Weights from file
    w_ih = load_w("w_ih")
    w_hh = load_w("w_hh")
    w_ho = load_w("w_ho")
    # 生成神经网络实例
    # create network
    net = NeuralNetwork(input_nodes, hidden_nodes_1,hidden_nodes_2, output_nodes, learning_rate, w_ih,w_hh,w_ho)
    return net

# 识别函数
# recognise function
def Recognize(N,net):
    # 输入的矩阵N是一个280x280的大矩阵，这里将其划分为100个28x28的区域，对每个区域求平均值，实现缩小图像以匹配神经网络
    # The input matrix N is a large matrix of 280x280. So I divide it into 100 28x28 regions,
    # averaging each region to achieve a zoomed-out image to match the neural network
    N_image = np.zeros((28, 28))
    for i in range(0, 280, 10):
        for j in range(0, 280, 10):
            x, y = i, j
            sum = 0
            for k in range(x, x + 10):
                for l in range(y, y + 10):
                    sum = sum + N[k, l]
            x_, y_ = int(i / 10), int(j / 10)
            N_image[x_, y_] = (sum / 100)
    # 将N_image转化为行向量
    # Converts N_image to row vectors
    N_image = np.reshape(N_image,(1,784))
    # 调用神经网络，返回识别结果
    # The neural network is invoked to return the recognition result
    outputs = net.query(N_image)
    result = np.argmax(outputs)
    return result

def showBackQuery(net,label):
    # 为传入的标签创建输出信号
    # create the output signals for this label
    targets = np.zeros(net.onodes) + 0.01
    # all_values[0] 是此记录的目标标签
    # all_values[0] is the target label for this record
    targets[label] = 0.99
    print(targets)
    # 得到图片数据
    # get image data
    image_data = net.backquery(targets)
    # 展示图片
    # plot image data
    matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()