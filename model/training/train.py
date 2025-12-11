import os

# Limit CPU thread parallelism to prevent 4000%+ CPU usage
# Must be set BEFORE importing torch
os.environ['OMP_NUM_THREADS'] = '4'        # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '4'        # Limit MKL (Intel Math Kernel Library)
os.environ['OPENBLAS_NUM_THREADS'] = '4'   # Limit OpenBLAS
os.environ['NUMEXPR_NUM_THREADS'] = '4'    # Limit NumExpr

import time
import argparse
import random
import gc
import torch

# Explicitly limit PyTorch intra-op parallelism
torch.set_num_threads(4)
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('..')
from utils.utils import load_config, save_checkpoint, load_checkpoint
from dataset import Words, get_dataset
from san_model import Backbone
from training import train, eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HYB Tree')
    parser.add_argument('--config', default='config.yaml', type=str, help='path to config file')
    parser.add_argument('--check', action='store_true', help='only for code check')
    args = parser.parse_args()

    if not args.config:
        print('please provide config yaml')
        exit(-1)

    """config"""
    params = load_config(args.config)

    """random seed"""
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])

    # Device selection: CUDA > CPU > MPS (MPS has tensor stride issues with this model)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        # Note: MPS backend has compatibility issues with complex tensor operations in this model
        # Use CPU for now until PyTorch MPS backend improves
        device = torch.device('cpu')

    params['device'] = device
    print(f'Using device: {device}')
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7
    params['words'] = words

    train_loader, eval_loader = get_dataset(params)

    model = Backbone(params)
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    model.name = f'{params["experiment"]}_{now}_Encoder-{params["encoder"]["net"]}_Decoder-{params["decoder"]["net"]}_' \
                 f'max_size-{params["image_height"]}-{params["image_width"]}'
    print(model.name)
    model = model.to(device)

    if args.check:
        writer = None
    else:
        writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

    optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                          eps=float(params['eps']), weight_decay=float(params['weight_decay']))

    # Initialize mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=params.get('use_amp', False))
    params['scaler'] = scaler

    start_epoch = 0
    min_score = -1
    min_step = 0

    if params['finetune']:
        print('loading pretrain model weight')
        print(f'pretrain model: {params["checkpoint"]}')
        training_state = load_checkpoint(model, optimizer, params['checkpoint'], scaler)
        start_epoch = training_state['epoch']
        min_score = training_state['best_score']
        min_step = training_state['min_step']
        print(f'Resuming from epoch {start_epoch}, best_score={min_score:.4f}, min_step={min_step}')

    if not args.check:
        if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
            os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
        os.system(f'cp {args.config} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')

    # Override base learning rate if specified in config (used by cosine LR schedule)
    if params.get('lr_override') is not None:
        params['lr'] = params['lr_override']
        print(f'NOTE: Overriding base learning rate to: {params["lr"]}')
    
    for epoch in range(start_epoch, params['epoches']):

        train_loss, train_word_score, train_node_score, train_expRate = train(params, model, optimizer, epoch, train_loader, writer=writer)
        if epoch > -1:
            eval_loss, eval_word_score, eval_node_score, eval_expRate = eval(params, model, epoch, eval_loader, writer=writer)

            print(f'Epoch: {epoch+1}  loss: {eval_loss:.4f}  word score: {eval_word_score:.4f}  struct score: {eval_node_score:.4f} '
                  f'ExpRate: {eval_expRate:.4f}')

            if eval_expRate >= min_score and not args.check:
                min_score = eval_expRate
                save_checkpoint(model, optimizer, eval_word_score, eval_node_score, eval_expRate, epoch+1,
                                optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'],
                                min_step=min_step, scaler=scaler)
                min_step = 0

            elif min_score != 0 and 'lr_decay' in params and params['lr_decay'] == 'step':

                min_step += 1

                if min_step > params['step_ratio']:
                    new_lr = optimizer.param_groups[0]['lr'] / params['step_decay']

                    if new_lr < params['lr'] / 1000:
                        print('lr is too small')
                        exit(-1)

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                    min_step = 0

        # Clean up memory after each epoch
        if writer:
            writer.flush()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()











