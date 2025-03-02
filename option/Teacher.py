import argparse
import json
import os
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='Automatic detection')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--iters_per_epoch', type=int, default=5000)
parser.add_argument('--finer_eval_step', type=int, default=100000)
parser.add_argument('--start_lr', default=0.0001, type=float, help='start learning rate')
parser.add_argument('--end_lr', default=0.000001, type=float, help='end learning rate')
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
parser.add_argument('--w_loss_L1', default=0.8, type=float, help='weight of loss L1')
parser.add_argument('--w_loss_SSIM', default=0.2, type=float, help='weight of loss SSIM')
parser.add_argument('--w_loss_Cr', default=0.05, type=float, help='weight of loss Cr')

parser.add_argument('--exp_dir', type=str, default='./experiment')
parser.add_argument('--model_name', type=str, default='THaze')
parser.add_argument('--saved_model_dir', type=str, default='saved_model')
parser.add_argument('--saved_data_dir', type=str, default='saved_data')
parser.add_argument('--dataset', type=str, default='Teacher')

opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
model_dir = os.path.join(dataset_dir, opt.model_name)

if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
    opt.saved_model_dir = os.path.join(model_dir, 'saved_model')
    opt.saved_data_dir = os.path.join(model_dir, 'saved_data')
    os.mkdir(opt.saved_model_dir)
    os.mkdir(opt.saved_data_dir)

with open(os.path.join(model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
