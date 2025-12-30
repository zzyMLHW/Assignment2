from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from NN_new import *
from nn_train_new import nn_train
from nn_forward import nn_forward
from nn_test import nn_test
from nn_predict import nn_predict
from nn_backward import nn_backward
from nn_applygradient_new import nn_applygradient
from function import *
import numpy as np
import pickle
import os


def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

filename = 'krkopt.data'
fr = open(filename)
arrayOLines = fr.readlines()
del arrayOLines[0]
numberOfLines = len(arrayOLines)
numberOfDimension = 6
data = np.zeros((numberOfLines, numberOfDimension), dtype=float)
label = np.zeros((numberOfLines, 2), dtype=float)
for index in range(len(arrayOLines)):
    line = arrayOLines[index]
    listFromLine = line.split(',')
    data[index, 0] = ord(listFromLine[0])-96
    data[index, 1] = ord(listFromLine[1]) - 48
    data[index, 2] = ord(listFromLine[2])-96
    data[index, 3] = ord(listFromLine[3]) - 48
    data[index, 4] = ord(listFromLine[4])-96
    data[index, 5] = ord(listFromLine[5]) - 48
    if listFromLine[6] == 'draw\n':
        label[index,:] = np.array([1,0])
    else:
        label[index,:] = np.array([0,1])

ratioTraining = 0.4
ratioValidation = 0.1
ratioTesting = 0.5
xTraining, xTesting, yTraining, yTesting = train_test_split(data, label, test_size=1 - ratioTraining, random_state=1)  # 随机分配数据集
xTesting, xValidation, yTesting, yValidation = train_test_split(xTesting, yTesting, test_size=ratioValidation / ratioTesting, random_state=1)

scaler = StandardScaler(copy=False)
scaler.fit(xTraining)
scaler.transform(xTraining)
scaler.transform(xTesting)
scaler.transform(xValidation)

alpha = 0.9
rho = 0.9

nn = NN(
    layer=[6, 20, 20, 20, 20, 2],
    active_function='sigmoid',
    batch_size = 100,
    learning_rate = 0.001,
    optimization_method='RMSProp_Nesterov',
    batch_normalization = 1,
    objective_function='Cross Entropy',
    rho=rho,
    alpha=alpha
)

epoch = 0
maxAccuracy = 0
totalAccuracy = []
totalCost = []
maxEpoches = 100
for epoch in range(maxEpoches):
    nn = nn_train(nn, xTraining, yTraining)
    totalCost.append(sum(nn.cost.values()) / len(nn.cost.values()))
    _, _, accuracy,_ = nn_test(nn, xValidation, yValidation)
    totalAccuracy.append(accuracy)
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        save_variable(nn, 'storedNN_Chess.npz')
    cost = totalCost[epoch]
    print('Epoch:',epoch)
    print('Accuracy:',accuracy)
    print('Cost:',totalCost[epoch])

# 绘制 Accuracy 和 Cost 曲线
epochs = range(1, len(totalAccuracy) + 1)
plt.figure(figsize=(12, 5))

# 绘制 Accuracy 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, totalAccuracy, 'b-', linewidth=2, label='Validation Accuracy')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy Curve (Chess Dataset)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 绘制 Cost 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, totalCost, 'r-', linewidth=2, label='Training Cost')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.title('Cost Curve (Chess Dataset)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('training_curves_chess.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形，避免影响后续绘图
print('训练曲线已保存为 training_curves_chess.png')

if os.path.exists('storedNN_Chess.npz'):
    storedNN = load_variable('storedNN_Chess.npz')
    wrongs, yPred, accuracy, yOutput = nn_test(storedNN, xTesting, yTesting)
    decisionValues = yOutput[:,0]
    print('Accuracy on Testset:', accuracy)
    truePositive,trueNegative,falsePositive,falseNegative = 0,0,0,0
    for i in range(len(yPred)):
        if np.argmax(yTesting[i,:])==0:
            if yPred[i]==0:
                truePositive+=1
            else:
                falseNegative+=1
        else:
            if yPred[i] == 0:
                falsePositive += 1
            else:
                trueNegative += 1
    print(truePositive, trueNegative, falsePositive, falseNegative)
    
    # 计算ROC曲线
    # decisionValues是draw（正类）的概率，值越大越可能是draw
    # 按决策值从高到低排序，这样阈值从高到低变化时，先预测决策值高的为正类
    totalScores = sorted(decisionValues, reverse=True)  # 从高到低排序
    index = sorted(range(len(decisionValues)), key=lambda i: decisionValues[i], reverse=True)
    
    # 获取真实标签（0表示draw正类，1表示not draw负类）
    labels = np.array([np.argmax(yTesting[i]) for i in index])

    # 计算正类和负类的总数
    numPositive = np.sum(labels == 0)  # 标签为0的样本数（draw）
    numNegative = np.sum(labels == 1)  # 标签为1的样本数（not draw）
    
    if numPositive == 0 or numNegative == 0:
        print('警告: 测试集中只有一个类别，无法绘制ROC曲线')
    else:
        # 初始化TPR和FPR数组
        tpr = np.zeros(len(labels) + 1)
        fpr = np.zeros(len(labels) + 1)
        
        # 从所有样本都被预测为负类开始（TP=0, FP=0, TPR=0, FPR=0）
        # 逐步降低阈值，将决策值高的样本先预测为正类
        for i in range(len(labels)):
            # 前i+1个样本（决策值最高的）被预测为正类
            # 计算这些样本中实际为正类和负类的数量
            tp = np.sum(labels[:i+1] == 0)  # 前i+1个样本中，实际为正类的数量
            fp = np.sum(labels[:i+1] == 1)  # 前i+1个样本中，实际为负类的数量
            
            tpr[i + 1] = tp / numPositive if numPositive > 0 else 0
            fpr[i + 1] = fp / numNegative if numNegative > 0 else 0
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve (Chess Dataset)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig('ROC.png', dpi=300, bbox_inches='tight')
        plt.show()
        print('ROC曲线已保存为 ROC.png')


