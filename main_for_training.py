# ANN-4
import numpy as np
import pickle
from ANN import NeuralNetwork as Net
import openpyxl as pxl
def save_w(w,name):
    f = open(name+'.pkl', 'wb')
    # 待写入数据
    datas = w
    # 写入
    data = pickle.dump(datas, f, -1)
    # 关闭文件
    f.close()
def load_w(name):
    f = open(name+'.pkl', 'rb')
    # 使用load的方法将数据从pkl文件中读取出来
    w = pickle.load(f)
    # 关闭文件
    f.close()
    return w

input_nodes = 784
hidden_nodes_1 = 80
hidden_nodes_2 = 40
output_nodes = 10

# learning rate
learning_rate = 0.1

# create network
net = Net(input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("E:\Program\AA-MyProtect\mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the network
epoch = 1  # 训练世代
i = 0
for e in range(epoch):
    for record in training_data_list:
        # 处理输入数据
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 创建target
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        net.train(inputs, targets)
        i+=1
        print("e{}-training {}".format(e + 1, i))
        pass
    pass
print("----train over----")

# load the mnist test data CSV file into a list
test_data_file = open("E:\Program\AA-MyProtect\mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the network
print(" ---Testing--- ")
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = net.query(inputs)
    label = np.argmax(outputs)

    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = np.asarray(scorecard)
print("正确率为：", scorecard_array.sum() / scorecard_array.size)
rate = scorecard_array.sum() / scorecard_array.size


#----------------------------------------------------------------
value = [hidden_nodes_1, hidden_nodes_2, rate]
book = pxl.load_workbook("Performance.xlsx")
sheet = book.active
sheet._current_row = int(sheet.max_row)
sheet.append(value)

book.save("Performance.xlsx")

try:
    f = open("MaxRate.txt", "r")
    max_rate = float(f.read())
    f.close()
except:
    max_rate = 0.0

if max_rate < rate:
    f = open("MaxRate.txt", "w")
    f.write(str(rate))
    save_w(net.w_ih, "w_ih")
    save_w(net.w_hh, "w_hh")
    save_w(net.w_ho, "w_ho")
    print("max_rate saved\n")

# 程序跑完提示音
import winsound
duration = 1000 # milliseconds
freq = 440 # Hz
winsound.Beep(freq, duration)