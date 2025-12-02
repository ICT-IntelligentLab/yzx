import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import *
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='DeepSC Advanced Training')
parser.add_argument('--datatrain-dir', default='train_25w', type=str)
parser.add_argument('--datatest-dir', default='test_2.5w', type=str)
parser.add_argument('--vocab-file', default='vocab/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/Bob1/Rayleigh', type=str)
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
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--grad-clip', default=1.0, type=float, help='Gradient clipping norm')
parser.add_argument('--patience', default=1000, type=int, help='Patience for early stopping')
parser.add_argument('--min-lr', default=1e-6, type=float, help='Minimum learning rate')
parser.add_argument('--label-smoothing', default=0.1, type=float, help='Label smoothing factor')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--warmup-epochs', default=5, type=int, help='Warmup epochs')
parser.add_argument('--accumulation-steps', default=1, type=int, help='Gradient accumulation steps')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AdvancedDeepSC(DeepSC):
    
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, 
                 src_len, trg_len, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__(num_layers, src_vocab_size, trg_vocab_size, 
                        src_len, trg_len, d_model, num_heads, dff, dropout_rate)
        
        self.encoder_dropout = nn.Dropout(dropout_rate)
        self.decoder_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, src, trg, src_mask, look_ahead_mask, n_var, channel='AWGN'):
        enc_output = self.encoder(src, src_mask)
        enc_output = self.encoder_dropout(enc_output)  # 添加dropout
        
        channel_enc_output = self.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)
        
        channels = Channels()
        if channel == 'None' or channel == 'NoChannel':
            Rx_sig = Tx_sig
        elif channel == 'AWGN':
            Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, Rician, or None")
        
        channel_dec_output = self.channel_decoder(Rx_sig)
        dec_output = self.decoder(trg, channel_dec_output, look_ahead_mask, src_mask)
        dec_output = self.decoder_dropout(dec_output)  # 添加dropout
        
        output = self.dense(dec_output)
        return output

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net, criterion):
   
    test_eur = EurDataset(args.datatest_dir)
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=4,
                                pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0

    progress = epoch / args.epochs
    
    min_snr = max(0, 5 - progress * 3)  
    max_snr = max(5, 15 - progress * 5) 

    with torch.no_grad():
        for sents in pbar:

            snr = np.random.uniform(min_snr, max_snr)
            noise_std = SNR_to_noise(snr)
            sents = sents.to(device)
            loss = val_step_advanced(net, sents, sents, noise_std, pad_idx,
                                   criterion, args.channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total/len(test_iterator)

def val_step_advanced(model, src, trg, n_var, pad, criterion, channel):
   
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    if hasattr(model, 'encoder_dropout'):
        enc_output = model.encoder_dropout(enc_output)
        
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'None' or channel == 'NoChannel':
        Rx_sig = Tx_sig 
    elif channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, Rician, or None")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    if hasattr(model, 'decoder_dropout'):
        dec_output = model.decoder_dropout(dec_output)
        
    pred = model.dense(dec_output)

    ntokens = pred.size(-1)
    
    if hasattr(criterion, 'smoothing'): 
        loss = criterion(pred.contiguous().view(-1, ntokens), 
                        trg_real.contiguous().view(-1))
        mask = (trg_real.contiguous().view(-1) != pad).type_as(loss)
        loss = (loss * mask).sum() / mask.sum()
    else: 
        loss = loss_function(pred.contiguous().view(-1, ntokens), 
                            trg_real.contiguous().view(-1), 
                            pad, criterion)
    return loss.item()

class LabelSmoothingCrossEntropy(nn.Module):
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train_epoch_advanced(epoch, args, net, mi_net, optimizer, criterion, scaler, scheduler):

    train_eur = EurDataset(args.datatrain_dir)
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=4,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)
    
    total_loss = 0
    num_batches = 0
    
    progress = epoch / args.epochs
    
    min_snr = max(0, 5 - progress * 3)  
    max_snr = max(5, 15 - progress * 5) 
    
    optimizer.zero_grad()
    
    for i, sents in enumerate(pbar):
        sents = sents.to(device)
        
        # 动态SNR
        snr = np.random.uniform(min_snr, max_snr)
        noise_std = SNR_to_noise(snr)

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents, noise_std, pad_idx, mi_opt, args.channel)
            loss = train_step_advanced(net, sents, sents, noise_std, pad_idx,
                                     optimizer, criterion, args.channel, mi_net, 
                                     args.grad_clip, scaler, args.accumulation_steps,
                                     batch_idx=i)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}; SNR: {:.1f}dB'.format(
                    epoch + 1, loss, mi, snr
                )
            )
        else:
            loss = train_step_advanced(net, sents, sents, noise_std, pad_idx,
                                     optimizer, criterion, args.channel, 
                                     grad_clip=args.grad_clip, scaler=scaler,
                                     accumulation_steps=args.accumulation_steps,
                                     batch_idx=i)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; SNR: {:.1f}dB'.format(
                    epoch + 1, loss, snr
                )
            )
        
        total_loss += loss
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    if scheduler is not None:
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_loss)
        else:
            scheduler.step()
    
    return avg_loss

def train_step_advanced(model, src, trg, n_var, pad, opt, criterion, channel, 
                       mi_net=None, grad_clip=1.0, scaler=None, accumulation_steps=1,
                       batch_idx=None):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    
    with autocast():
        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
        
        enc_output = model.encoder(src, src_mask)
        if hasattr(model, 'encoder_dropout'):
            enc_output = model.encoder_dropout(enc_output)
            
        channel_enc_output = model.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)

        if channel == 'None' or channel == 'NoChannel':
            Rx_sig = Tx_sig
        elif channel == 'AWGN':
            Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, Rician, or None")

        channel_dec_output = model.channel_decoder(Rx_sig)
        dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        if hasattr(model, 'decoder_dropout'):
            dec_output = model.decoder_dropout(dec_output)
            
        pred = model.dense(dec_output)
        
        ntokens = pred.size(-1)
        
        if hasattr(criterion, 'smoothing'): 
            loss = criterion(pred.contiguous().view(-1, ntokens), 
                            trg_real.contiguous().view(-1))
            mask = (trg_real.contiguous().view(-1) != pad).type_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else: 
            loss = loss_function(pred.contiguous().view(-1, ntokens), 
                                trg_real.contiguous().view(-1), 
                                pad, criterion)

        if mi_net is not None:
            mi_net.eval()
            joint, marginal = sample_batch(Tx_sig, Rx_sig)
            mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
            loss_mine = -mi_lb
            loss = loss + 0.0009 * loss_mine

   
    scaler.scale(loss).backward()
    
    if batch_idx is not None and (batch_idx + 1) % accumulation_steps == 0:
        # 梯度裁剪
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

    return loss.item()

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

def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'training_curves.png'))
    plt.close()

if __name__ == '__main__':
    # setup_seed(42) 
    args = parser.parse_args()
    vocab = json.load(open(args.vocab_file, 'r', encoding='utf-8'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ 定义模型和优化器 """
    deepsc = AdvancedDeepSC(args.num_layers, num_vocab, num_vocab,
                           num_vocab, num_vocab, args.d_model, args.num_heads,
                           args.dff, args.dropout).to(device)
    
    mi_net = Mine().to(device)
    
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        print(f"使用标签平滑: {args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
    
    optimizer = torch.optim.AdamW(deepsc.parameters(), 
                                 lr=args.lr, betas=(0.9, 0.98), eps=1e-8, 
                                 weight_decay=args.weight_decay)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=args.lr)
    

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = GradScaler()
    initNetParams(deepsc)
    start_epoch = 0
    if args.resume:
        try:
            start_epoch = load_checkpoint(args.resume, deepsc, optimizer, mi_net, mi_opt)
        except Exception as e:
            print(f"加载checkpoint失败: {e}")
            print("将从头开始训练")
            start_epoch = 0

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    best_acc = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"开始高级优化训练，从epoch {start_epoch + 1} 到 {args.epochs}")
    print(f"高级配置:")
    print(f"  - 标签平滑: {args.label_smoothing}")
    print(f"  - Dropout: {args.dropout}")
    print(f"  - 梯度累积步数: {args.accumulation_steps}")
    print(f"  - 梯度裁剪: {args.grad_clip}")
    print(f"  - 早停耐心: {args.patience}")
    print(f"  - 最小学习率: {args.min_lr}")
    
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        train_loss = train_epoch_advanced(epoch, args, deepsc, mi_net, optimizer, criterion, scaler, scheduler)
        train_losses.append(train_loss)
        
        # 验证
        avg_acc = validate(epoch, args, deepsc, criterion)
        val_losses.append(avg_acc)
        
        # 早停机制
        if avg_acc < best_acc:
            best_acc = avg_acc
            patience_counter = 0
            checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_best.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': deepsc.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mi_net': mi_net.state_dict() if mi_net else None,
                'mi_opt': mi_opt.state_dict() if mi_opt else None,
                'best_acc': best_acc,
                'scheduler': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_file)
            print(f" 新的最佳验证损失: {best_acc:.5f}")
        else:
            patience_counter += 1
            print(f"验证损失未改善，耐心计数: {patience_counter}/{args.patience}")
        
        # 检查早停
        if patience_counter >= args.patience:
            print(f"早停触发！在epoch {epoch + 1}停止训练")
            break
        
        # 定期保存checkpoint和训练曲线
        if (epoch + 1) % 10 == 0:
            checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_{epoch + 1:03d}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': deepsc.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mi_net': mi_net.state_dict() if mi_net else None,
                'mi_opt': mi_opt.state_dict() if mi_opt else None,
                'best_acc': best_acc,
                'scheduler': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_file)
            
            # 绘制训练曲线
            plot_training_curves(train_losses, val_losses, args.checkpoint_path)
        
        epoch_time = time.time() - start
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1} 完成，耗时: {epoch_time:.2f}s, 训练损失: {train_loss:.5f}, 验证损失: {avg_acc:.5f}, 学习率: {current_lr:.2e}')
    
    # 最终保存训练曲线
    plot_training_curves(train_losses, val_losses, args.checkpoint_path)
    print(f"训练完成！最佳验证损失: {best_acc:.5f}")
