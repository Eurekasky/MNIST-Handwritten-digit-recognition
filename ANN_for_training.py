#neural Network class definition
import numpy as np
import scipy.special as spe
class NeuralNetwork:
    def __init__(self,inputnodes,hiddennodes_1,hiddennodes_2,outputnodes,learning_rate):
        # 设置节点数
        self.inodes = inputnodes
        self.hnodes_1 = hiddennodes_1
        self.hnodes_2 = hiddennodes_2
        self.onodes = outputnodes
        # 设置学习率
        self.lr  = learning_rate
        # 初始化权重
        self.w_ih = np.random.normal(0.0,pow(self.hnodes_1,-0.5),(self.hnodes_1,self.inodes))
        self.w_hh = np.random.normal(0.0,pow(self.hnodes_2,-0.5),(self.hnodes_2,self.hnodes_1))
        self.w_ho = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes_2))
        # 激活函数
        self.active_fun = lambda x: spe.expit(x)
        pass

    def train(self,inputs_list,targets_list):
        # 把输入转换成二维矩阵
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T

        # 正向传播计算
        hidden_1_inputs = np.dot(self.w_ih,inputs)
        hidden_1_outputs = self.active_fun(hidden_1_inputs)
        hidden_2_inputs = np.dot(self.w_hh,hidden_1_outputs)
        hidden_2_outputs = self.active_fun(hidden_2_inputs)
        out_inputs = np.dot(self.w_ho,hidden_2_outputs)
        out_outputs = self.active_fun(out_inputs)

        # 反向传播计算
        output_errors = targets - out_inputs
        hidden_2_errors = np.dot(self.w_ho.T,output_errors)
        hidden_1_errors = np.dot(self.w_hh.T,hidden_2_errors)
        # 更新权重
        self.w_ho += self.lr * np.dot((output_errors * out_outputs *(1.0-out_outputs)),np.transpose(hidden_2_outputs))
        self.w_hh += self.lr * np.dot((hidden_2_errors * hidden_2_outputs * (1.0-hidden_2_outputs)),np.transpose(hidden_1_outputs))
        self.w_ih += self.lr * np.dot((hidden_1_errors * hidden_1_outputs *(1.0-hidden_1_outputs)),np.transpose(inputs))
        pass

    def query(self,inputs_list):
        # 把输入转换成二维矩阵
        inputs = np.array(inputs_list,ndmin=2).T
        # 正向传播计算
        hidden_1_inputs = np.dot(self.w_ih,inputs)
        hidden_1_outputs = self.active_fun(hidden_1_inputs)
        hidden_2_inputs = np.dot(self.w_hh,hidden_1_outputs)
        hidden_2_outputs = self.active_fun(hidden_2_inputs)
        out_inputs = np.dot(self.w_ho,hidden_2_outputs)
        out_outputs = self.active_fun(out_inputs)
        return out_outputs