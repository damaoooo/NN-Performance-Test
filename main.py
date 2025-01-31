import argparse
import time
from utils import *
from compile_model_transformer import MNIST_Transformer
from compile_model_cnn import MNIST_ResNet

# debug模式下的默认参数
DEBUG_ARGS = {
    'train': False,
    'test': True,
    'device': 'cpu',
    'model': 'transformer',
    'batch_size': 64,
    'epochs': 2,  # debug模式下减少epoch数
    'lr': 0.01,
    'max_train_batches': 5,  # debug模式下限制训练batch数
    'max_test_batches': 5,   # debug模式下限制测试batch数
}

def get_model(model_name):
    """获取指定的模型"""
    if model_name == "transformer":
        return MNIST_Transformer()
    elif model_name == "cnn":
        return MNIST_ResNet()
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

def get_device_model(model, device='cpu'):
    """根据指定的设备返回处理后的模型和设备"""
    # device = torch.device(device)  # 创建设备对象
    if device == "npu":
        return npu_model(model), device
    elif device == "cuda":
        return cuda_model(model), device
    elif device == "mps":
        return mps_model(model), device
    else:  # cpu
        return model.to(device), device

def main():
    parser = argparse.ArgumentParser(description='MNIST模型训练和测试程序')
    parser.add_argument('--debug', action='store_true', default=True, help='是否使用debug模式')
    parser.add_argument('--train', action='store_true', help='是否进行训练')
    parser.add_argument('--test', action='store_true', help='是否进行测试')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'npu', 'mps'], help='运行设备')
    parser.add_argument('--model', type=str, default='cnn', choices=['transformer', 'cnn'], help='模型类型')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--max_train_batches', type=int, default=None, help='最大训练batch数')
    parser.add_argument('--max_test_batches', type=int, default=None, help='最大测试batch数')

    args = parser.parse_args()

    # 如果是debug模式，使用DEBUG_ARGS中的默认值
    if args.debug:
        print("\n=== Debug模式已启用 ===")
        # 创建一个字典来存储命令行参数的值
        cmd_args = vars(args)
        # 更新未在命令行中指定的参数
        for key, value in DEBUG_ARGS.items():
            cmd_args[key] = value
        # 将更新后的字典转换回Namespace对象
        args = argparse.Namespace(**cmd_args)
        print("当前参数设置:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        print("================\n")

    # 如果既没有指定训练也没有指定测试，则两者都执行
    if not args.train and not args.test:
        args.train = True
        args.test = True

    # 获取数据加载器
    train_loader, test_loader = get_mnist_data_loaders(batch_size=args.batch_size)

    # 创建模型和获取设备
    model = get_model(args.model)
    model, device = get_device_model(model, args.device)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    if args.train:
        print(f"\n开始训练 - 模型: {args.model}, 设备: {args.device}")
        train_start_time = time.time()
        train_model(model, train_loader, optimizer, device,
                   num_epochs=args.epochs, 
                   max_batches=args.max_train_batches)
        train_time = time.time() - train_start_time
        print(f"训练完成 - 总用时: {train_time:.2f} 秒")

    # 测试
    if args.test:
        print(f"\n开始测试 - 模型: {args.model}, 设备: {args.device}")
        test_loss, accuracy, inference_time = test_model(model, test_loader, device,
                                                       max_batches=args.max_test_batches)
        print(f"测试完成 - 推理用时: {inference_time:.2f} 秒")

if __name__ == "__main__":
    main() 