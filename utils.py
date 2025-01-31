import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import tqdm

def npu_model(model):
    from intel_npu_acceleration_library import compile
    from intel_npu_acceleration_library.compiler import CompilerConfig

    import intel_npu_acceleration_library

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

def mps_model(model):
    # 苹果M芯片的MPS
    import torch.backends.mps as mps
    mps.is_available()
    mps.is_built()
    device = torch.device("mps")
    model.to(device)
    return model

def cuda_model(model):
    # 英伟达GPU
    device = torch.device("cuda")
    model.to(device)
    return model

def get_mnist_data_loaders(batch_size=64):
    """
    获取MNIST数据集的数据加载器
    
    Args:
        batch_size: 批次大小，默认为64
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        transform=transform, 
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, device, epoch, max_batches=None):
    """
    训练一个epoch
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 运行设备
        epoch: 当前epoch数
        max_batches: 最大训练批次数，如果为None则训练整个数据集
    """
    batch_times = []
    progress_bar = tqdm.tqdm(train_loader, leave=False)
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        if max_batches and batch_idx >= max_batches:
            break
            
        start_time = time.time()
        
        # 将数据移动到指定设备
        if device != 'npu':
            data = data.to(device)
            target = target.to(device)
        
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        if len(batch_times) > 10:
            batch_times = batch_times[-10:]
        avg_time = sum(batch_times) / len(batch_times)
        batches_per_second = 1.0 / avg_time
        
        progress_bar.set_description(f'Epoch {epoch} - {batches_per_second:.2f} batch/s')

def train_model(model, train_loader, optimizer, device, num_epochs=10, max_batches=5):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 运行设备
        num_epochs: 训练轮数，默认为10
        max_batches: 每个epoch最大训练批次数，默认为5
    """
    total_start_time = time.time()
    for epoch in tqdm.tqdm(range(num_epochs), desc='Epochs'):
        epoch_start_time = time.time()
        train_epoch(model, train_loader, optimizer, device, epoch, max_batches=max_batches)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} 用时: {epoch_time:.2f} 秒")
        print(f"平均每个batch用时: {epoch_time/max_batches:.4f} 秒")
    
    total_time = time.time() - total_start_time
    print(f"总训练用时: {total_time:.2f} 秒")
    print(f"总batch数: {num_epochs * max_batches}")
    print(f"平均每个batch用时: {total_time/(num_epochs * max_batches):.4f} 秒")
    return total_time

def test_model(model, test_loader, device, max_batches=None):
    """
    测试模型
    
    Args:
        model: 要测试的模型
        test_loader: 测试数据加载器
        device: 运行设备
        max_batches: 最大测试批次数，如果为None则测试所有数据
        
    Returns:
        test_loss: 测试损失
        accuracy: 准确率
        inference_time: 推理时间
    """
    test_loss = 0
    correct = 0
    batch_times = []
    counter = 0
    inference_start_time = time.time()
    
    with torch.no_grad():
        test_progress = tqdm.tqdm(test_loader, desc='Testing', leave=True)
        for data, target in test_progress:
            counter += 1
            if max_batches and counter > max_batches:
                break
                
            # 将数据移动到指定设备
            if device != 'npu':
                data = data.to(device)
                target = target.to(device)
                
            start_time = time.time()
            
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            if len(batch_times) > 10:
                batch_times = batch_times[-10:]
            avg_time = sum(batch_times) / len(batch_times)
            batches_per_second = 1.0 / avg_time
            
            current_acc = 100. * correct / ((test_progress.n + 1) * test_loader.batch_size)
            test_progress.set_description(
                f'Testing - {batches_per_second:.2f} batch/s - Acc: {current_acc:.2f}%'
            )
            
    inference_time = time.time() - inference_start_time
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    print(f"Inference time: {inference_time:.2f} seconds")
    
    return test_loss, accuracy, inference_time