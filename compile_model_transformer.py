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
from transformers import BertModel, BertConfig
import torchvision.transforms as transforms

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

class MNIST_Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 配置BERT模型参数
        self.config = BertConfig(
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=785,
            type_vocab_size=1,
            vocab_size=256
        )
        
        self.bert = BertModel(self.config)
        
        # 修改图像预处理层
        self.pixel_to_patch = torch.nn.Sequential(
            torch.nn.Conv2d(1, 768, kernel_size=1),  # [batch_size, 768, 28, 28]
            torch.nn.Flatten(2),  # [batch_size, 768, 784]
        )
        
        # 单独的LayerNorm层，用于序列维度
        self.layer_norm = torch.nn.LayerNorm(768)
        
        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # 将图像转换为序列 [batch_size, 768, 784]
        x = self.pixel_to_patch(x)
        
        # 转置为BERT期望的输入格式 [batch_size, 784, 768]
        x = x.transpose(1, 2)
        
        # 对每个位置应用LayerNorm
        x = self.layer_norm(x)
        
        # 添加[CLS]标记的位置嵌入
        cls_tokens = torch.zeros(batch_size, 1, 768).to(x.device)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 创建注意力掩码
        attention_mask = torch.ones(batch_size, 785).to(x.device)
        
        # 通过BERT
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)
        
        # 使用[CLS]标记的输出进行分类
        cls_output = outputs.last_hidden_state[:, 0]
        
        # 分类
        logits = self.classifier(cls_output)
        
        return logits

if __name__ == "__main__":

    # Define a NN module
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = MNIST_Transformer()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # compiled_model = npu_model(model)
    compiled_model = model

    # # Train the model
    # for epoch in tqdm.tqdm(range(10), desc='Epochs'):
    #     batch_times = []
    #     progress_bar = tqdm.tqdm(train_loader, leave=False)
    #     for batch_idx, (data, target) in enumerate(progress_bar):
    #         start_time = time.time()
            
    #         output = compiled_model(data)
    #         loss = torch.nn.functional.cross_entropy(output, target)
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
    test_loss = 0
    correct = 0
    batch_times = []
    counter = 0
    inference_start_time = time.time()
    with torch.no_grad():
        test_progress = tqdm.tqdm(test_loader, desc='Testing', leave=True)
        for data, target in test_progress:
            counter += 1
            if counter > 5:
                break
            start_time = time.time()
            
            output = compiled_model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 计算速度
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            if len(batch_times) > 10:
                batch_times = batch_times[-10:]
            avg_time = sum(batch_times) / len(batch_times)
            batches_per_second = 1.0 / avg_time
            
            # 更新进度条，同时显示当前准确率
            current_acc = 100. * correct / ((test_progress.n + 1) * test_loader.batch_size)
            test_progress.set_description(
                f'Testing - {batches_per_second:.2f} batch/s - Acc: {current_acc:.2f}%'
            )
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time
    print(f"Inference time: {inference_time:.2f} seconds")


    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)")



    