import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi, NoamOpt
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser(description='DeepSC Resume Training')
parser.add_argument('--datatrain-dir', default='train_25w', type=str)
parser.add_argument('--datatest-dir', default='test_2.5w', type=str)
parser.add_argument('--vocab-file', default='vocab/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/Bob1/Rayleigh/resume', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=52, type=int)  
parser.add_argument('--MIN-LENGTH', default=10, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=600, type=int)
parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume from')
parser.add_argument('--start-epoch', default=0, type=int, help='Manual epoch number (useful on restarts)')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net):
    test_eur = EurDataset(args.datatest_dir)
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=4,
                                pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx,
                             criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total/len(test_iterator)


def train(epoch, args, net, mi_net=None):
    train_eur = EurDataset(args.datatrain_dir)
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=4,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(0), SNR_to_noise(8), size=(1))

    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents,noise_std[0], pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )


def load_checkpoint(checkpoint_path, model, optimizer=None, mi_net=None, mi_opt=None):
    if os.path.isfile(checkpoint_path):
        print(f"=> 加载checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型权重
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> 加载优化器状态")
        
        # 加载MI网络状态
        if mi_net is not None and 'mi_net' in checkpoint:
            mi_net.load_state_dict(checkpoint['mi_net'])
            print("=> 加载MI网络状态")
        
        if mi_opt is not None and 'mi_opt' in checkpoint:
            mi_opt.load_state_dict(checkpoint['mi_opt'])
            print("=> 加载MI优化器状态")
        
        # 获取epoch信息
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f"=> 从epoch {start_epoch}继续训练")
            return start_epoch
        else:
            print("=> 未找到epoch信息，从epoch 0开始")
            return 0
    else:
        raise FileNotFoundError(f"=> 未找到checkpoint '{checkpoint_path}'")


def save_checkpoint(state, filename):
    """保存checkpoint，包含训练状态"""
    torch.save(state, filename)
    print(f"=> 保存checkpoint: {filename}")


if __name__ == '__main__':
    # setup_seed(10)
    args = parser.parse_args()
    vocab = json.load(open(args.vocab_file, 'r', encoding='utf-8'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ 定义模型和优化器 """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)
    
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    opt = NoamOpt(args.d_model, 1, 4000, optimizer)
    
    # 初始化网络参数
    initNetParams(deepsc)

    # 继续训练逻辑
    start_epoch = args.start_epoch
    if args.resume:
        try:
            start_epoch = load_checkpoint(args.resume, deepsc, optimizer, mi_net, mi_opt)
        except Exception as e:
            print(f"加载checkpoint失败: {e}")
            print("将从头开始训练")
            start_epoch = 0

    # 创建保存目录
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    record_acc = 10
    best_acc = record_acc
    
    print(f"开始训练，从epoch {start_epoch + 1} 到 {args.epochs}")
    
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        
        # 训练一个epoch
        train(epoch, args, deepsc, mi_net)
        
        # 验证
        avg_acc = validate(epoch, args, deepsc)
        
        # 保存最佳模型
        if avg_acc < best_acc:
            best_acc = avg_acc
            checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_best.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': deepsc.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mi_net': mi_net.state_dict() if mi_net else None,
                'mi_opt': mi_opt.state_dict() if mi_opt else None,
                'best_acc': best_acc,
            }, checkpoint_file)
            print(f"新的最佳准确率: {best_acc:.5f}")
        
        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_{epoch + 1:03d}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': deepsc.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mi_net': mi_net.state_dict() if mi_net else None,
                'mi_opt': mi_opt.state_dict() if mi_opt else None,
                'best_acc': best_acc,
            }, checkpoint_file)
        
        epoch_time = time.time() - start
        print(f'Epoch {epoch + 1} 完成，耗时: {epoch_time:.2f}s, 验证损失: {avg_acc:.5f}')
    
    print(f"训练完成！最佳验证损失: {best_acc:.5f}")
