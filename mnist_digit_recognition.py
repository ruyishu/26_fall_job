import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# 数据加载与预处理
def get_data_loaders(batch_size=64):
    """
    获取MNIST数据集的训练集和测试集加载器。
    
    参数:
        batch_size (int): 每个批次的图片数量，默认为64。
    
    返回:
        train_loader, test_loader: 训练集和测试集的加载器。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化到均值为0.1307，标准差为0.3081
    ])

    # 加载训练集和测试集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # DataLoader用于批量处理数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader

# CNN 模型定义
class CNN(nn.Module):
    """
    基于卷积神经网络的MNIST手写数字识别模型
    """
    def __init__(self):
        super(CNN, self).__init__()
        # 定义第一个卷积层：输入通道1（灰度图），输出通道32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 定义第二个卷积层：输入通道32，输出通道64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 最大池化层：2x2 池化
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层：64*7*7 映射到128维
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 输出层：将128维映射到10个类别（数字0-9）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层 + 激活
        x = F.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x

# 模型训练函数
def train(model, device, train_loader, optimizer, epoch):
    """
    训练模型并输出训练过程中损失值。
    
    参数:
        model: CNN 模型
        device: 运行设备（CPU 或 GPU）
        train_loader: 训练集加载器
        optimizer: 优化器
        epoch: 当前训练的轮数
    """
    model.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 将数据加载到设备（GPU或CPU）
        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 前向传播
        loss = F.cross_entropy(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# 模型评估函数
def evaluate(model, device, test_loader):
    """
    测试模型并输出在测试集上的准确率。
    
    参数:
        model: CNN 模型
        device: 运行设备（CPU 或 GPU）
        test_loader: 测试集加载器
    """
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # 选取最大概率的类别
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

# 主函数，整合训练与评估过程
def main():
    # 参数设置
    batch_size = 64
    epochs = 5
    learning_rate = 0.001

    # 设备选择：CPU 或 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_loader, test_loader = get_data_loaders(batch_size)

    # 初始化模型和优化器
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和评估模型
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader)

    # 保存训练好的模型
    torch.save(model.state_dict(), "mnist_cnn_model.pth")

# 执行主函数
if __name__ == '__main__':
    main()
