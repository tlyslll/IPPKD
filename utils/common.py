
import os
import sys
import shutil
import time, datetime
import logging
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils


'''record configurations'''
class record_config():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.output_dir = Path(args.output_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.output_dir)

        config_dir = self.output_dir / 'config.txt'
        #if not os.path.exists(config_dir):
        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')


def get_logger(file_path):
    logger = logging.getLogger('gal')
    #logger = logging.getLogger('gal')
    #print(file_path)
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    #loggerr=logging.getLogger(file_path)
    # if not os.path.exists(file_path):
    #     print(file_path+" is not exists")
    #     with open(file_path,"w") as f:
    #         f.write("1")
    #         f.close()
    file_path=file_path.replace(":","_")
    file_handler = logging.FileHandler(file_path)
    #print(file_handler)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

#label smooth
class Dual_CrossEntropyLoss(nn.Module):
    def __init__(self,alpha=0.1,beta=0.1):
        super(Dual_CrossEntropyLoss, self).__init__()
        self.alpha=alpha
        self.beta1=beta
        self.beta2=(1-beta)
        self.logsoftmax=nn.LogSoftmax(dim=1)
    def forward(self, logits, target):
        # 计算交叉熵
        ce_log_probs=self.logsoftmax(logits)
        lr_log_probs=self.logsoftmax(self.alpha+logits)
        target=torch.zeros_like(ce_log_probs).scatter_(1,target.unsqueeze(1),1)
        #print(1-target)
        ce_loss=(-target*ce_log_probs).mean(0).sum()
        lr_loss=(-(1-target)*lr_log_probs*1e-3).mean(0).sum()
        loss=self.beta2*ce_loss+self.beta1*lr_loss
        return loss

class CE_L0(nn.Module):
    def __init__(self,beta=1e-3):
        super(CE_L0, self).__init__()
        #self.alpha=alpha
        self.beta1=beta
        self.beta2=(1-beta)
        self.logsoftmax=nn.LogSoftmax(dim=1)
        self.norm_zero=torch.tensor(0,dtype=torch.float).cuda()
    def forward(self, logits, target,model):
        # 计算交叉熵
        #self.norm_zero=self.norm_zero.cuda()
        self.norm_zero = self.norm_zero.cuda()
        ce_log_probs=self.logsoftmax(logits)
        target=torch.zeros_like(ce_log_probs).scatter_(1,target.unsqueeze(1),1)
        ce_loss=(-target*ce_log_probs).mean(0).sum()
        for param in model.parameters():
            self.norm_zero+=torch.sum(param==0)
        #lr_loss=(-(1-target)*lr_log_probs*1e-3).mean(0).sum()
        loss=self.beta2*ce_loss+self.beta1*self.norm_zero
        return loss

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    #print(targets.size())
    loss = (-targets * log_probs).mean(0).sum()
    #l
    return loss

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.pt')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pt')
        shutil.copyfile(filename, best_filename)


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def progress_bar(current, total, msg=None):
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

    TOTAL_BAR_LENGTH = 65.
    last_time = time.time()
    begin_time = last_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

import math
class CosLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, len_iter, last_epoch=-1, epochs_warmup = 3, eta_min = 0, verbose=False):
        self.T_max = T_max
        self.len_iter = len_iter
        self.e_wup = epochs_warmup
        self.eta_min = eta_min
        super(CosLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        epoch = (self.last_epoch)//self.len_iter
        lr = [(0.5 * base_lr * (1 + math.cos(math.pi * (epoch - self.e_wup) / (self.T_max - self.e_wup)))) for base_lr in self.base_lrs]
        if epoch<self.e_wup:
            step = self.last_epoch % self.len_iter
            lr = [e * float(1 + step + epoch * self.len_iter) / (self.e_wup * self.len_iter) for e in lr]
        lr = [e if e >= self.eta_min else self.eta_min for e in lr]
        return lr
