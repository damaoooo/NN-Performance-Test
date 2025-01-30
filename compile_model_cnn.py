from intel_npu_acceleration_library import compile
from intel_npu_acceleration_library.compiler import CompilerConfig
from sklearn.metrics import r2_score
import intel_npu_acceleration_library
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
import sys
import tqdm
import time  # 添加time模块导入
# Import mnist dataset
from torchvision import datasets, transforms
import torchvision.models as models

def npu_model(model):
    # Compile the model
    print("Compile the model for the NPU")
    if sys.platform == "win32":

        # Windows do not support torch.compile
        print(
            "Windows do not support torch.compile, fallback to intel_npu_acceleration_library.compile"
        )
        compiler_conf = CompilerConfig()
        compiled_model = intel_npu_acceleration_library.compile(model, compiler_conf)
    else:
        compiled_model = torch.compile(model, backend="npu")
    return compiled_model

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

    # Define a NN module
    model = MNIST_ResNet()
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Compile the model
    compiled_model = npu_model(model)
    # compiled_model = model

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # # Train the model
    # for epoch in tqdm.tqdm(range(10), desc='Epochs'):
    #     batch_times = []
    #     progress_bar = tqdm.tqdm(train_loader, leave=False)
    #     for batch_idx, (data, target) in enumerate(progress_bar):
    #         start_time = time.time()
            
    #         output = compiled_model(data)
    #         loss = torch.nn.functional.nll_loss(output, target)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         batch_time = time.time() - start_time
    #         batch_times.append(batch_time)
            
    #         # 计算并更新每秒批次数
    #         if len(batch_times) > 10:  # 使用最近10个批次的平均值
    #             batch_times = batch_times[-10:]
    #         avg_time = sum(batch_times) / len(batch_times)
    #         batches_per_second = 1.0 / avg_time
            
    #         # 更新进度条描述
    #         progress_bar.set_description(f'Epoch {epoch} - {batches_per_second:.2f} batch/s')

    # Test the model
    inference_start_time = time.time()
    test_loss = 0
    correct = 0
    batch_times = []
    with torch.no_grad():
        test_progress = tqdm.tqdm(test_loader, desc='Testing', leave=True)
        for data, target in test_progress:
            start_time = time.time()
            
            output = compiled_model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 计算速度
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            if len(batch_times) > 10:  # 保留最近10个批次的时间
                batch_times = batch_times[-10:]
            avg_time = sum(batch_times) / len(batch_times)
            batches_per_second = 1.0 / avg_time
            
            # 更新进度条，显示速度和当前准确率
            current_acc = 100. * correct / ((test_progress.n + 1) * test_loader.batch_size)
            test_progress.set_description(
                f'Testing - {batches_per_second:.2f} batch/s - Acc: {current_acc:.2f}%'
            )
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time
    print(f"Inference time: {inference_time:.2f} seconds")
    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)")




    