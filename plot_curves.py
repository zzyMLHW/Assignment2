import matplotlib.pyplot as plt
import pickle
import os

def load_variable(filename):
    """加载保存的变量"""
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

def plot_training_curves(data_file='training_data.pkl', save_path='training_curves.png'):
    """
    从文件读取训练数据并绘制曲线
    
    参数:
        data_file: 训练数据文件路径（默认: 'training_data.pkl'）
        save_path: 保存图片的路径（默认: 'training_curves.png'）
    """
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f'错误: 文件 {data_file} 不存在！')
        print('请先运行 testMNIST.py 生成训练数据文件。')
        return
    
    # 加载数据
    print(f'正在从 {data_file} 加载数据...')
    training_data = load_variable(data_file)
    
    totalAccuracy = training_data['totalAccuracy']
    totalCost = training_data['totalCost']
    epochs = training_data['epochs']
    
    print(f'成功加载数据: {len(epochs)} 个 epoch')
    
    # 绘制曲线
    plt.figure(figsize=(12, 5))
    
    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, totalAccuracy, 'b-', linewidth=2, label='Validation Accuracy')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Curve', fontsize=14, fontweight='bold')
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
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'曲线图已保存为 {save_path}')
    
    # 显示图片
    plt.show()
    
    # 打印统计信息
    print('\n训练统计信息:')
    print(f'最终 Accuracy: {totalAccuracy[-1]:.4f}')
    print(f'最高 Accuracy: {max(totalAccuracy):.4f} (Epoch {epochs[totalAccuracy.index(max(totalAccuracy))]})')
    print(f'最终 Cost: {totalCost[-1]:.6f}')
    print(f'最低 Cost: {min(totalCost):.6f} (Epoch {epochs[totalCost.index(min(totalCost))]})')

if __name__ == '__main__':
    # 默认使用 training_data.pkl 文件
    plot_training_curves()
    
    # 如果需要指定其他文件，可以这样调用:
    # plot_training_curves(data_file='your_data_file.pkl', save_path='your_output.png')

