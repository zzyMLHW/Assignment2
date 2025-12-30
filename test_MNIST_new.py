from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from NN_new import *
from nn_train_new import nn_train
from nn_test import nn_test
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from nn_forward import nn_forward
from nn_predict import nn_predict
from nn_backward import nn_backward
from nn_applygradient_new import nn_applygradient
from function import sigmoid, softmax

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

#Training
dir_path = './MNIST/train'
if not os.path.exists(dir_path):
    print(f"Error: Directory {dir_path} not found.")
else:
    file_ls = os.listdir(dir_path)

data = np.zeros((60000, 784), dtype=float)
label = np.zeros((60000, 10), dtype=float)
flag = 0

try:
    for dir in file_ls:
        files = os.listdir(dir_path+'/'+dir)
        for file in files:
            filename = dir_path+'/'+dir+'/'+file
            img = mpimg.imread(filename)
            data[flag,:] = np.reshape(img, -1)/255
            label[flag,int(dir)] = 1.0
            flag+=1
except Exception as e:
    print("数据加载部分跳过或出错 (如果你是在测试代码逻辑请忽略):", e)

ratioTraining = 0.95
xTraining, xValidation, yTraining, yValidation = train_test_split(data, label, test_size=1 - ratioTraining, random_state=0)

model_filename = 'storedNN_Nesterov.npz' 

if os.path.exists(model_filename):
    print(f"Loading existing model from {model_filename}...")
    nn = load_variable(model_filename)
else:
    print("Initializing new NN with RMSProp_Nesterov...")
    nn = NN(
        layer=[784, 400, 169, 49, 10], 
        batch_normalization=1, 
        active_function='relu', 
        batch_size=50, 
        learning_rate=0.0005, 
        optimization_method='RMSProp_Nesterov', 
        objective_function='Cross Entropy',
        rho=0.9,   # RMSProp decay
        alpha=0.9  # Nesterov momentum
    )

epoch = 0
maxAccuracy = 0
totalAccuracy = []
totalCost = []
maxEpoch = 100

while epoch < maxEpoch:
    epoch += 1
    # 训练
    nn = nn_train(nn, xTraining, yTraining)
    
    # 记录 Cost
    current_cost = sum(nn.cost.values()) / len(nn.cost.values())
    totalCost.append(current_cost)
    
    # 验证集测试
    wrongs, predictedLabel, accuracy, yOutput = nn_test(nn, xValidation, yValidation)
    totalAccuracy.append(accuracy)
    
    # 保存最佳模型
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        save_variable(nn, model_filename) # 使用新的文件名保存
    
    print(f'Epoch: {epoch} | Accuracy: {accuracy:.4f} | Cost: {current_cost:.4f}')

# 绘制 Accuracy 和 Cost 曲线
epochs = range(1, len(totalAccuracy) + 1)
plt.figure(figsize=(12, 5))

# 绘制 Accuracy 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, totalAccuracy, 'b-', linewidth=2, label='Validation Accuracy')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy Curve (RMSProp + Nesterov)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 绘制 Cost 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, totalCost, 'r-', linewidth=2, label='Training Cost')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.title('Cost Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('training_curves_nesterov.png', dpi=300, bbox_inches='tight')
plt.show()
print('训练曲线已保存为 training_curves_nesterov.png')

# 保存训练数据
training_data = {
    'totalAccuracy': totalAccuracy,
    'totalCost': totalCost,
    'epochs': list(range(1, len(totalAccuracy) + 1))
}
save_variable(training_data, 'training_data_nesterov.pkl')

#Testing
dir_path = './MNIST/test'
xTesting = np.zeros((10000, 784), dtype=float)
yTesting = np.zeros((10000, 10), dtype=float)

# 加载测试数据逻辑 (略微加了防错)
if os.path.exists(dir_path):
    file_ls = os.listdir(dir_path)
    flag = 0
    try:
        for dir in file_ls:
            files = os.listdir(dir_path+'/'+dir)
            for file in files:
                filename = dir_path+'/'+dir+'/'+file
                img = mpimg.imread(filename)
                xTesting[flag,:] = np.reshape(img, -1)/255
                yTesting[flag,int(dir)] = 1.0
                flag+=1
    except Exception as e:
        print("测试集加载出错:", e)

    # 加载刚才训练好的最佳模型
    if os.path.exists(model_filename):
        storedNN = load_variable(model_filename)
        wrongs, predictedLabel, accuracy, yOutput = nn_test(storedNN, xTesting, yTesting)
        print('Accuracy on Test set:', accuracy)
        confusionMatrix = np.zeros((10,10),dtype=int)
        for i in range(len(predictedLabel)):
            trueLabel = np.argmax(yTesting[i,:])
            confusionMatrix[trueLabel,predictedLabel[i]]+=1
        print('The Confusion Matrix is:\n', confusionMatrix)
else:
    print("Test directory not found.")