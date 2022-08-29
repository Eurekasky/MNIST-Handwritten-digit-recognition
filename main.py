import tkinter as tk
import numpy as np
import ANN

# 绘画函数。两个作用：在canvas画布上画椭圆来显示笔迹；在矩阵N中置1来保存图片数据
# Painting functions. Two roles: draw an ellipse on the canvas to
# show handwriting; Place 1 in matrix N to save the picture data
def paint(event):
    global N
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    cv.create_oval(x1, y1, x2, y2, fill="white")
    k = 10
    for x in range(event.x-k,event.x+k):
        for y in range(event.y-k,event.y+k):
            if 0<=x and x<280 and y<280 and 0<y :
                N[y,x] = 0.99

# 识别函数。将矩阵N传入net得到识别结果，并更新结果到Entry控件。最后刷新笔迹
# Identify function. Pass the matrix N into net to get the
# recognition result and update the result to the Entry control.
# Finally refresh the Canva and matrix N
def Recgonize():
    global N
    global net
    global result
    global txt
    # 如果N为空矩阵，则输出空格/If N is an empty matrix, a space is output
    if N.sum()==0:
        result=" "
    else:
        result = ANN.Recognize(N,net)
    N = np.zeros((280,280))  # 重新初始化数组N
    cv.delete(tk.ALL) #清空画布上所有元素
    s = txt.get()     #先读取Entry控件的文本，然后在尾部追加最新结果
    s += str(result)
    txt.set(s)
    print(s)

# 空格按钮的对应函数
# The corresponding function of the space button
def Blank():
    global txt
    s = txt.get()
    s += " "
    txt.set(s)

# 删除按钮的对应函数
# Deletes the corresponding function of the button
def Delete():
    global txt
    s = txt.get()
    new_s = s[:-1]
    txt.set(str(new_s))

##################################-------- main --------###########################################
# 以下是主程序部分
# The following is the main program section

# 创建矩阵N来存储手写图形的数据。空白为0，有图像置0.99
# Create a matrix N to store the data for the
# handwritten graph. The blank is 0, and the image is set to 0.99
N = np.zeros((280,280))
result = -1
# 初始化神经网络
# Initialize the neural network
net = ANN.InitializeANN()

# 创建窗口
# Build the main window
root = tk.Tk()
root.geometry("285x460+700+100")
root.title("手写数字识别")
# 创建标签1
label_1 = tk.Label(root,text="Hello World",height=3)
label_1.grid(row=0,column=0,columnspan=2,sticky="EW")

# 创建Entry控件，用于输出识别结果
# Build an Entry control that outputs recognition results
txt = tk.StringVar()
txt.set("识别结果：")
output= tk.Entry(root,textvariable=txt,font=("Calibri",11))
output.grid(row=1,column=0,columnspan=2,sticky="EW",ipady=6)

# 创建按钮
# Bulid Buttons
b_recgonize = tk.Button(root,text="识别",command=Recgonize,bg="#D2B48C")
b_recgonize.grid(row=2,column=0,columnspan=2,sticky="EW")
b_blank = tk.Button(root,text="空格",command=Blank,bg="#C0C0C0")
b_blank.grid(row=3,column=0,sticky="EW")
b_delete = tk.Button(root,text="删除",command=Delete,bg="#C0C0C0")
b_delete.grid(row=3,column=1,sticky="EW")
# 创建标签2
label_2 = tk.Label(root,text="@ANNx4",font=("Calibri",8),height=1)
label_2.grid(row=4,column=0,columnspan=2,sticky="W")
# 创建画布
# Build the canvas
cv = tk.Canvas(root, width=280, height=280,bg="black")
cv.grid(row=5,column=0,columnspan=2)
cv.bind("<B1-Motion>", paint)

root.mainloop()

# 这个是调用反向查询的语句。将前面窗口部分代码全部注释掉即可使用
# This is the statement that calls the BackQuery method.
# Comment out all the code in the previous window and use it

#ANN.showBackQuery(net,9)