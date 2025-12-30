import matplotlib.pyplot as plt
import pickle
import os


def load_variable(filename):
    """加载保存的变量"""
    with open(filename, 'rb') as f:
        r = pickle.load(f)
    return r


def compare_curves(file1='training_data.pkl', file2='training_data_nesterov.pkl', save_path='comparison_curves.png'):
    """
    读取两个训练数据文件并绘制对比曲线
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file1):
        print(f'错误: 文件 {file1} 不存在！')
        return
    if not os.path.exists(file2):
        print(f'错误: 文件 {file2} 不存在！')
        return

    # 2. 加载数据
    print(f'正在加载 {file1} (Baseline: Momentum)...')
    data1 = load_variable(file1)

    print(f'正在加载 {file2} (RMSProp + Nesterov)...')
    data2 = load_variable(file2)

    # 提取数据 1 (Baseline - Momentum)
    acc1 = data1['totalAccuracy']
    cost1 = data1['totalCost']
    epochs1 = data1['epochs']

    # 提取数据 2 (RMSProp + Nesterov)
    acc2 = data2['totalAccuracy']
    cost2 = data2['totalCost']
    epochs2 = data2['epochs']

    # 3. 绘制曲线
    plt.figure(figsize=(14, 6))

    # --- 左图：Accuracy 对比 ---
    plt.subplot(1, 2, 1)
    # 绘制第一条线 (蓝色虚线) - Momentum
    plt.plot(epochs1, acc1, 'b--', linewidth=1.5, alpha=0.7, label='Momentum (Baseline)')
    # 绘制第二条线 (红色实线) - RMSProp + Nesterov
    plt.plot(epochs2, acc2, 'r-', linewidth=2, label='RMSProp + Nesterov')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')

    # --- 右图：Cost 对比 ---
    plt.subplot(1, 2, 2)
    # 绘制第一条线 (蓝色虚线) - Momentum
    plt.plot(epochs1, cost1, 'b--', linewidth=1.5, alpha=0.7, label='Momentum (Baseline)')
    # 绘制第二条线 (红色实线) - RMSProp + Nesterov
    plt.plot(epochs2, cost2, 'r-', linewidth=2, label='RMSProp + Nesterov')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Training Cost Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n对比曲线图已保存为 {save_path}')

    # 显示图片
    plt.show()

    # 4. 打印数值对比表格
    print('=' * 60)
    print(f"{'Metric':<20} | {'Momentum':<20} | {'RMSProp+Nesterov':<20}")
    print('-' * 66)
    print(f"{'Max Accuracy':<20} | {max(acc1):.4f}               | {max(acc2):.4f}")
    print(f"{'Final Cost':<20} | {cost1[-1]:.4f}               | {cost2[-1]:.4f}")
    print('=' * 60)


if __name__ == '__main__':
    compare_curves()