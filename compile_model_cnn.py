from utils import *

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models


class MNIST_ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的ResNet18模型
        self.model = models.resnet101()
        
        # 修改第一层卷积以接受单通道输入（MNIST是灰度图像）
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改最后的全连接层以输出10个类别（MNIST的数字0-9）
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, 10)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # 获取数据加载器
    train_loader, test_loader = get_mnist_data_loaders(batch_size=64)
    
    # 创建模型
    model = MNIST_ResNet()
    
    # 编译模型
    compiled_model = npu_model(model)
    # compiled_model = model

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    train_model(compiled_model, train_loader, optimizer, num_epochs=10)

    # 测试模型
    test_loss, accuracy, inference_time = test_model(compiled_model, test_loader)




    