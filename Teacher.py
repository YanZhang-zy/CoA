import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import optim, nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from loss import SSIM, ContrastLoss
from data import RESIDE_Dataset, TestDataset
from metric import psnr, ssim
from model import Teacher
from option.Teacher import opt


start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def train(teacher_net, loader_train_1, loader_test, optim, criterion):
    losses = []
    loss_log = {'L1': [], 'SSIM': [], 'Cr': [], 'total': []}
    loss_log_tmp = {'L1': [], 'SSIM': [], 'Cr': [], 'total': []}
    psnr_log = []

    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    loader_train_iter_1 = iter(loader_train_1)

    for step in range(start_step + 1, steps + 1):
        teacher_net.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        try:
            x, y = next(loader_train_iter_1)
        except StopIteration:
            loader_train_iter_1 = iter(loader_train_1)
            x, y = next(loader_train_iter_1)

        x = x.to(opt.device, non_blocking=True)
        y = y.to(opt.device, non_blocking=True)

        teacher_out = teacher_net(x)

        loss_L1 = criterion[0](teacher_out[0], y) if opt.w_loss_L1 > 0 else 0
        loss_SSIM = (1 - criterion[1](teacher_out[0], y)) if opt.w_loss_SSIM > 0 else 0
        loss_Cr = criterion[2](teacher_out[0], y, x) if opt.w_loss_Cr > 0 else 0

        loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_SSIM * loss_SSIM + opt.w_loss_Cr * loss_Cr

        loss.backward()
        optim.step()
        optim.zero_grad()

        losses.append(loss.item())
        loss_log_tmp['L1'].append(loss_L1.item() if opt.w_loss_L1 > 0 else 0)
        loss_log_tmp['SSIM'].append(loss_SSIM.item() if opt.w_loss_SSIM > 0 else 0)
        loss_log_tmp['Cr'].append(loss_Cr.item() if opt.w_loss_Cr > 0 else 0)
        loss_log_tmp['total'].append(loss.item())

        print(f'\rloss:{loss.item():.5f} | L1:{opt.w_loss_L1 * loss_L1.item():.5f} | SSIM:{opt.w_loss_SSIM * loss_SSIM.item():.5f} | Cr:{opt.w_loss_Cr * loss_Cr.item():.5f}  | step :{step}/{steps} | lr :{lr :.9f} | time_used :{(time.time() - start_time) / 60 :.1f}', end='', flush=True)

        if step % len(loader_train_1) == 0:
            loader_train_iter_1 = iter(loader_train_1)
            for key in loss_log.keys():
                loss_log[key].append(np.average(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = []
            np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)

        if (step % opt.iters_per_epoch == 0 and step <= opt.finer_eval_step) or (
                step > opt.finer_eval_step and (step - opt.finer_eval_step) % (5 * len(loader_train_1)) == 0):
            if step > opt.finer_eval_step:
                epoch = opt.finer_eval_step // opt.iters_per_epoch + (step - opt.finer_eval_step) // (
                        5 * len(loader_train_1))
            else:
                epoch = int(step / opt.iters_per_epoch)
            with torch.no_grad():
                ssim_eval, psnr_eval = test(teacher_net, loader_test)

            log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f} | lr:{lr:.12f}'
            print(log)
            with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)
            state_dict = teacher_net.state_dict()
            if 'module' in list(state_dict.keys())[0]:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                state_dict = new_state_dict
            if psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                print(f'model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')
                saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pth')
                torch.save(state_dict, saved_best_model_path)
            saved_single_model_path = os.path.join(opt.saved_model_dir, str(epoch) + '.pth')
            torch.save(state_dict, saved_single_model_path)
            loader_train_iter_1 = iter(loader_train_1)
            np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
            np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    for i, (inputs, targets, hazy_name) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            H, W = inputs.shape[2:]
            inputs = pad_img(inputs, 4)
            pred = net(inputs)[0].clamp(0, 1)
            pred = pred[:, :, :H, :W]
        ssim_tmp = ssim(pred, targets).item()
        psnr_tmp = psnr(pred, targets)
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)

    return np.mean(ssims), np.mean(psnrs)


def set_seed_torch(seed=2024):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    set_seed_torch(2024)

    train_dir_1 = './data/THaze/train'
    train_set_1 = RESIDE_Dataset(train_dir_1, True, 256, '.jpg')

    test_dir = './data/THaze/test'
    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'))

    loader_train_1 = DataLoader(dataset=train_set_1, batch_size=24, shuffle=True, num_workers=8)
    loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

    teacher_net = Teacher()
    teacher_net = teacher_net.to(opt.device)

    epoch_size = len(loader_train_1)
    print("epoch_size: ", epoch_size)
    if opt.device == 'cuda':
        teacher_net = torch.nn.DataParallel(teacher_net)
        cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in teacher_net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))
    print("------------------------------------------------------------------")

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(SSIM().to(opt.device))
    criterion.append(ContrastLoss(ablation=False))

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, teacher_net.parameters()), lr=opt.start_lr,
                           betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    train(teacher_net, loader_train_1, loader_test, optimizer, criterion)
