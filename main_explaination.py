import argparse  # 导入解析命令行参数的模块
import os  # 导入操作系统路径的模块
import random  # 导入随机数生成的模块
import shutil  # 导入文件和文件夹操作的模块
import time  # 导入时间相关的模块
import warnings  # 导入警告处理的模块
from enum import Enum  # 导入枚举类型的模块

import torch  # 导入PyTorch深度学习框架
import torch.backends.cudnn as cudnn  # 导入PyTorch的CUDA加速库
import torch.distributed as dist  # 导入PyTorch的分布式训练模块
import torch.multiprocessing as mp  # 导入PyTorch的多进程模块
import torch.nn as nn  # 导入神经网络模块
import torch.nn.parallel  # 导入模型并行计算的模块
import torch.optim  # 导入优化器模块
import torch.utils.data  # 导入数据加载模块
import torch.utils.data.distributed  # 导入分布式数据加载模块
import torchvision.datasets as datasets  # 导入PyTorch的数据集模块
import torchvision.models as models  # 导入PyTorch的模型模块
import torchvision.transforms as transforms  # 导入数据预处理模块
from torch.optim.lr_scheduler import StepLR  # 导入学习率调整策略
from torch.utils.data import Subset  # 导入数据子集模块
from torch.utils.tensorboard import SummaryWriter # 导入tensorboard写入器

# 根据项目组织目录改变以下值
path_tiny_imagenet_200 = '/data/bitahub/tiny-imagenet-200/' #xxxx/xxxx/
# # 定义TensorBoard写入器
writer = SummaryWriter(log_dir='/output/logs')


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))  # 获取可用的模型名称列表

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')  # 创建解析命令行参数的解析器
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='数据集路径（默认：imagenet）')  # 添加一个命令行参数，用于指定数据集路径
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='模型架构：' +
                         ' | '.join(model_names) +
                         '（默认：resnet18）')  # 添加一个命令行参数，用于指定模型架构

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='数据加载的工作进程数（默认：4）')  # 添加一个命令行参数，用于指定数据加载的工作进程数
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='总共运行的训练周期数')  # 添加一个命令行参数，用于指定训练总周期数
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='手动设置的训练周期数（在重新启动时有用）')  # 添加一个命令行参数，用于手动设置训练周期数
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='小批量大小（默认：256），当使用数据并行或分布式数据并行时，这是当前节点上所有 GPU 的总批量大小')  # 添加一个命令行参数，用于指定小批量大小
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='初始学习率', dest='lr')  # 添加一个命令行参数，用于指定初始学习率
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='动量')  # 添加一个命令行参数，用于指定动量
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='权重衰减（默认：1e-4）',
                    dest='weight_decay')  # 添加一个命令行参数，用于指定权重衰减
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='打印频率（默认：10）')  # 添加一个命令行参数，用于指定打印频率
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='最新检查点的路径（默认：无）')  # 添加一个命令行参数，用于指定最新检查点的路径
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='对验证集上的模型进行评估')  # 添加一个命令行参数，用于指定是否在验证集上评估模型
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='使用预训练模型')  # 添加一个命令行参数，用于指定是否使用预训练模型
parser.add_argument('--world-size', default=-1, type=int,
                    help='分布式训练的节点数')  # 添加一个命令行参数，用于指定分布式训练的节点数
parser.add_argument('--rank', default=-1, type=int,
                    help='分布式训练的节点排名')  # 添加一个命令行参数，用于指定分布式训练的节点排名
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='用于设置分布式训练的 URL')  # 添加一个命令行参数，用于指定分布式训练的 URL
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')  # 添加一个命令行参数，用于指定分布式训练的后端
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')  # 添加一个命令行参数，用于指定训练的随机种子
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')  # 添加一个命令行参数，用于指定要使用的 GPU 的 ID
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='使用多进程分布式训练，在每个节点上启动 N 个进程，每个进程有 N 个 GPU。这是使用 PyTorch 进行单节点或多节点数据并行训练的最快方式')  # 添加一个命令行参数，用于指定是否使用多进程分布式训练
parser.add_argument('--dummy', action='store_true', help="使用虚假数据进行基准测试")  # 添加一个命令行参数，用于指定是否使用虚假数据进行基准测试

best_acc1 = 0  # 初始化最佳准确率为0


def main():
    args = parser.parse_args()  # 解析命令行参数并将其存储在args对象中

    if args.seed is not None:
        random.seed(args.seed)  # 设置随机种子
        torch.manual_seed(args.seed)  # 设置PyTorch的随机种子
        cudnn.deterministic = True  # 设置CUDNN为确定性模式，确保结果可复现
        cudnn.benchmark = False  # 禁用CUDNN的基准模式，以确保训练结果的一致性
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')  # 发出警告，提醒用户选择了种子训练会影响训练速度并可能导致意外行为

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')  # 发出警告，提醒用户选择了特定的GPU，这将完全禁用数据并行性

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])  # 从环境变量中获取分布式训练的节点数

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed  # 根据节点数和是否启用多进程分布式训练设置分布式训练标志

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()  # 获取可用的GPU数量
    else:
        ngpus_per_node = 1  # 如果没有GPU可用，则将GPU数量设置为1

    if args.multiprocessing_distributed:
        # 由于每个节点有ngpus_per_node个进程，因此需要相应调整总的world_size
        args.world_size = ngpus_per_node * args.world_size
        # 使用torch.multiprocessing.spawn启动分布式进程：主要的工作进程函数为main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # 直接调用main_worker函数
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1  # 声明全局变量best_acc1

    args.gpu = gpu  # 将GPU编号存储在args.gpu中

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))  # 打印使用的GPU编号

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])  # 从环境变量中获取分布式训练的排名

        if args.multiprocessing_distributed:
            # 对于多进程分布式训练，rank需要是所有进程中的全局排名
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)  # 初始化分布式进程组

    # 创建模型
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)  # 使用预训练模型
        # # 修改最后一层的输出维度 
        ##TO_DO:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 200)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()  # 创建新的模型

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')  # 如果没有GPU可用，则打印警告信息，使用CPU将会很慢
    elif args.distributed:
        # 对于多进程分布式训练，DistributedDataParallel构造函数应始终设置单个设备范围，否则DistributedDataParallel将使用所有可用设备
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # 当每个进程和每个DistributedDataParallel使用单个GPU时，需要根据当前节点的总GPU数自行划分批次大小
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # 如果未设置device_ids，则DistributedDataParallel将批次大小分配给所有可用的GPU
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel将批次大小分配给所有可用的GPU
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 定义损失函数(criterion)、优化器(optimizer)和学习率调度器(scheduler)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 设置学习率调度器，每30个epoch将学习率衰减10倍

    # # 定义TensorBoard写入器
    #

    # 将模型写入TensorBoard
    writer.add_graph(model, torch.zeros([1, 3, 64, 64]))

    # 可选地从检查点恢复训练-checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # 将要加载的模型映射到指定的单个GPU
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1可能来自于不同GPU的检查点
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
                


    # 加载数据 #TO_DO:
    if args.dummy:  # 如果使用虚假数据
        print("=> Dummy data is used!")  # 打印提示信息
        # 创建虚假数据集
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:  # 如果使用真实数据
        traindir = os.path.join(args.data, 'train')  # 训练集路径
        valdir = os.path.join(args.data, 'val')  # 验证集路径
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化参数
                                        std=[0.229, 0.224, 0.225])

        # 训练集数据预处理
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                # transforms.RandomResizedCrop(224),  # 随机裁剪
                # transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ToTensor(),  # 转换为张量
                normalize,  # 归一化
            ]))
        
        #TO_DO:
        # 注意此段代码中的文件路径要与组合成项目后的路径符合，main.py与所要操作的文件的目录关系
        # 读取 wnids.txt 文件中的标签列表
        with open(path_tiny_imagenet_200+'wnids.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # 读取 val_annotations.txt 文件中的标签信息
        with open(path_tiny_imagenet_200+'val/val_annotations.txt', 'r') as f:
            val_annotations = [line.strip().split('\t') for line in f.readlines()]

        # 将每个样本的标签修正为对应标签在列表中的索引
        corrected_annotations = []
        for annotation in val_annotations:
            #print(annotation)
            filename, label = annotation[0], annotation[1]
            corrected_label = labels.index(label)
            corrected_annotations.append((filename, corrected_label, annotation[2], annotation[3], annotation[4], annotation[5]))

        # 将修正后的标签保存到新文件中
        with open(path_tiny_imagenet_200+'val/val_annotations_new.txt', 'w') as f:
            for annotation in corrected_annotations:
                f.write('\t'.join([str(x) for x in annotation]) + '\n')

        # 更改修改后的文件的名称为原来文件的名称
        os.rename(path_tiny_imagenet_200+'val/val_annotations.txt', path_tiny_imagenet_200+'val/val_annotations_original.txt')
        os.rename(path_tiny_imagenet_200+'val/val_annotations_new.txt', path_tiny_imagenet_200+'val/val_annotations.txt')
        print("=> val labels have already been right")

        # 验证集数据预处理
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                # transforms.Resize(256),  # 调整大小
                # transforms.CenterCrop(224),  # 中心裁剪
                transforms.ToTensor(),  # 转换为张量
                normalize,  # 归一化
            ]))


    if args.distributed:  # 如果是分布式训练
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  # 在训练集上应用分布式采样器
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)  # 在验证集上应用分布式采样器
    else:  # 如果不是分布式训练
        train_sampler = None  # 训练采样器为空
        val_sampler = None  # 验证采样器为空

    train_loader = torch.utils.data.DataLoader(  # 加载训练集
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)  # 在训练集上应用采样器

    val_loader = torch.utils.data.DataLoader(  # 加载验证集
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)  # 在验证集上应用采样器

    if args.evaluate:  # 如果只是评估模型
        validate(val_loader, model, criterion, args)  # 在验证集上评估模型
        return  # 返回

    for epoch in range(args.start_epoch, args.epochs):  # 对于每个epoch
        if args.distributed:  # 如果是分布式训练
            train_sampler.set_epoch(epoch)  # 设置训练采样器的epoch

        train(train_loader, model, criterion, optimizer, epoch, device, args)  # 训练一个epoch

        acc1 = validate(val_loader, model, criterion, args)  # 在验证集上评估模型

        scheduler.step()  # 更新学习率

        is_best = acc1 > best_acc1  # 判断是否是最佳准确率
        best_acc1 = max(acc1, best_acc1)  # 记录最佳准确率

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):  # 如果不是多进程分布式训练或者是主进程
            save_checkpoint({  # 保存检查点
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')  # 用于记录每个批次的训练时间的平均值
    data_time = AverageMeter('Data', ':6.3f')  # 用于记录每个批次的数据加载时间的平均值
    losses = AverageMeter('Loss', ':.4e')  # 用于记录每个批次的损失值的平均值
    top1 = AverageMeter('Acc@1', ':6.2f')  # 用于记录每个批次的Top-1准确率的平均值
    top5 = AverageMeter('Acc@5', ':6.2f')  # 用于记录每个批次的Top-5准确率的平均值
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))  # 用于显示训练进度的进度条

    # 切换到训练模式
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # 计算数据加载时间
        data_time.update(time.time() - end)

        # 将数据移动到与模型相同的设备上
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 计算输出
        output = model(images)
        loss = criterion(output, target)

        # 计算准确率并记录损失
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # 计算梯度并执行SGD步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算经过的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
        
        # 将训练集的Loss和精度写入TensorBoard
        if i % args.print_freq == 0:
            writer.add_scalar('Train/Loss', losses.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Train/Acc@1', top1.avg, epoch * len(train_loader) + i)
            writer.add_scalar('Train/Acc@5', top5.avg, epoch * len(train_loader) + i)

def validate(val_loader, model, criterion, args):
    # 定义一个名为run_validate的辅助函数，用于执行验证过程
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                # 如果GPU可用且指定了GPU编号，则将图像移动到相应的GPU设备上
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                # 如果支持MPS加速，则将图像和目标移动到MPS设备上
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                # 如果GPU可用，则将目标移动到相应的GPU设备上
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # 计算输出
                output = model(images)
                loss = criterion(output, target)

                # 计算准确率并记录损失
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # 将损失和准确率记录到 TensorBoard 中
                writer.add_scalar('val_loss', loss.item(), i)
                writer.add_scalar('val_acc1', acc1[0], i)
                writer.add_scalar('val_acc5', acc5[0], i)

                # 计算经过的时间
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    # 用于记录每个批次的验证时间的平均值
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    # 用于记录每个批次的损失值的平均值
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    # 用于记录每个批次的Top-1准确率的平均值
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    # 用于记录每个批次的Top-5准确率的平均值
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    # 用于显示验证进度的进度条
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # 切换到评估模式
    model.eval()

    run_validate(val_loader)  # 运行验证过程，传入验证数据加载器

    if args.distributed:
        top1.all_reduce()  # 在分布式训练中，对top1进行全局归约操作
        top5.all_reduce()  # 在分布式训练中，对top5进行全局归约操作

    # 如果是分布式训练，并且剩余的样本数量大于当前进程数乘以每个进程的样本数，
    # 则需要对剩余样本进行额外的验证，以确保所有样本都被评估
    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        # 创建剩余样本的子数据集
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        # 创建剩余样本的数据加载器
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))  # 运行剩余样本的验证过程，传入剩余样本数据加载器和基础进度值

    progress.display_summary()  # 显示验证过程的总结信息

    return top1  # 返回Top-1准确率

#TO_DO:
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # 保存模型状态到文件
    torch.save(state, filename)
    if is_best:
        # 如果当前模型是最佳模型，则复制保存一个副本作为最佳模型
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    # 摘要类型的枚举类
    NONE = 0  # 不计算摘要
    AVERAGE = 1  # 计算平均值
    SUM = 2  # 计算总和
    COUNT = 3  # 计数

class AverageMeter(object):
    """计算和存储平均值和当前值"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name  # 名称
        self.fmt = fmt  # 格式化字符串
        self.summary_type = summary_type  # 摘要类型
        self.reset()  # 重置计数器

def reset(self):
    # 重置计数器的初始值
    self.val = 0  # 当前值
    self.avg = 0  # 平均值
    self.sum = 0  # 总和
    self.count = 0  # 计数

def update(self, val, n=1):
    # 更新计数器的值
    self.val = val  # 当前值
    self.sum += val * n  # 累加总和
    self.count += n  # 增加计数
    self.avg = self.sum / self.count  # 计算平均值

def all_reduce(self):
    # 对所有进程的计数器进行全局归约操作
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)  # 创建张量用于存储总和和计数
    dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)  # 执行全局归约操作
    self.sum, self.count = total.tolist()  # 更新本地计数器的值
    self.avg = self.sum / self.count  # 更新平均值

def __str__(self):
    # 返回计数器的字符串表示形式
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

def summary(self):
    # 返回计数器的摘要字符串表示形式
    fmtstr = ''
    if self.summary_type is Summary.NONE:
        fmtstr = ''
    elif self.summary_type is Summary.AVERAGE:
        fmtstr = '{name} {avg:.3f}'
    elif self.summary_type is Summary.SUM:
        fmtstr = '{name} {sum:.3f}'
    elif self.summary_type is Summary.COUNT:
        fmtstr = '{name} {count:.3f}'
    else:
        raise ValueError('invalid summary type %r' % self.summary_type)
    
    return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        # 初始化进度显示器
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)  # 批次格式化字符串
        self.meters = meters  # 用于显示的度量器列表
        self.prefix = prefix  # 前缀字符串

    def display(self, batch):
        # 显示当前批次的进度
        entries = [self.prefix + self.batch_fmtstr.format(batch)]  # 批次信息
        entries += [str(meter) for meter in self.meters]  # 度量器信息
        print('\t'.join(entries))  # 打印信息

    def display_summary(self):
        # 显示摘要信息
        entries = [" *"]  # 摘要信息列表
        entries += [meter.summary() for meter in self.meters]  # 获取每个度量器的摘要信息
        print(' '.join(entries))  # 打印摘要信息

    def _get_batch_fmtstr(self, num_batches):
        # 获取批次格式化字符串
        num_digits = len(str(num_batches // 1))  # 计算批次数的位数
        fmt = '{:' + str(num_digits) + 'd}'  # 格式化字符串模板
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'  # 返回格式化字符串

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # 计算指定topk值下的预测准确率
    with torch.no_grad():
        maxk = max(topk)  # topk中的最大值
        batch_size = target.size(0)  # 批次大小

        #TO_DO:
        _, pred = output.topk(maxk, 1, True, True)  # 获取前k个预测结果
        pred = pred.t()  # 转置预测结果矩阵
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 比较预测结果和目标值是否相等

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # 计算前k个预测结果的准确数量
            res.append(correct_k.mul_(100.0 / batch_size))  # 计算准确率并添加到结果列表中
        return res

if __name__ == '__main__':
    main()
    writer.close()
