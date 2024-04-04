import os
import functools
from gpt2.utils import *
from dataset import get_dataset
from gpt2.sample import generate_text

# module for multi GPU training
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
# DistributedDataParallel: Use single-machine multi-GPU 
from torch.nn.parallel import DistributedDataParallel as DDP
# FullyShardedDataParallel: Use multi-GPU training on a single-machine or multi-machine 
# when the data and model cannot fit on one GPU.
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (size_based_auto_wrap_policy)

def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
        dist.destroy_process_group()

def ddp_train(rank, world_size, model, config):
        print('\nLoading dateset..')
        dataset = get_dataset(config)

        print('Running DDP on rank {rank}.')
        setup(rank, world_size)

        model.to(rank)
        model = DDP(model, device_ids=[rank])
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)
        
        model_file = latest_weights_file_path(config) if config.preload=='latest' else get_weights_file_path(config, config.preload) if config.preload else None        
        init_epoch = 0
        print(f'\nModel Training on {rank}..')
        if model_file:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                state = torch.load(model_file, map_location)
                init_epoch = state['epoch']+1
                print(f'>> Resuming model training from epoch no. {init_epoch}')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
        else:
                print('>> No model to preload, starting from scratch.')

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss_function = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing).to(rank)
        for epoch in range(init_epoch, config.epochs):
                torch.cuda.empty_cache()
                model.train()
                batch_count = 0
                if config.batch_data_while_training:
                        print(f'\n>>> epoch: {epoch+1}')
                for batch in dataset['train_dataset']:
                        input = batch['input'].to(rank) # shape: (batch, seq)
                        label = batch['label'].to(rank) # shape: (batch, seq)

                        out, _ = model(input) # (batch, seq, vocab_size)

                        # loss
                        loss = loss_function(out.contiguous().view(-1, out.size(-1)), torch.eye(config.vocab_size)[label.view(-1)])
                        loss.backward()
                        
                        # optimization
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        batch_count+=1
                        if config.batch_data_while_training:
                                print(f'>>> batch: {batch_count}, loss: {loss.item():6.3f}')
                        # break

                # to save the model instance at end of every epoch
                file = get_weights_file_path(config,f'{epoch:03d}')
                dist.barrier()
                torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, f = file)

                if not config.batch_data_while_training:
                        print(f'\n>>> epoch: {epoch+1}, loss: {loss.item():6.3f}')

                # validation step
                if config.validation_step_while_training:
                        if epoch % config.validation_step_frequency==0:
                                print('>> Validation step..')
                                generate_text(model, config, rank, text=dataset['val_dataset']['text'], unconditional=True)

        optimizer.zero_grad()

        end.record()
        torch.cuda.synchronize()
        print(f'\nTraining complete with total time {start.elapsed_time(end)/1000:.3f} sec.')

        cleanup()

def init_dpp(world_size, model, config):
        assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"

        mp.spawn(ddp_train,
                 args=(world_size, model, config),
                 nprocs=world_size,
                 join=True)
        
def fsdp_train(rank, world_size, model, config):
        print('Running FSDP on rank {rank}.')
        setup(rank, world_size)

        print('\nLoading dateset..')
        dataset = get_dataset(config, fsdp=True, rank=rank, world_size=world_size)
        my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
        torch.cuda.set_device(rank)

        model.to(rank)
        model = FSDP(model, fsdp_auto_wrap_policy=my_auto_wrap_policy)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)

        model_file = latest_weights_file_path(config) if config.preload=='latest' else get_weights_file_path(config, config.preload) if config.preload else None        
        init_epoch = 0
        print(f'\nModel Training on {rank}..')
        if model_file:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                state = torch.load(model_file, map_location)
                init_epoch = state['epoch']+1
                print(f'>> Resuming model training from epoch no. {init_epoch}')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
        else:
                print('>> No model to preload, starting from scratch.')

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss_function = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing, reduction='mean').to(rank)
        for epoch in range(init_epoch, config.epochs):
                torch.cuda.empty_cache()
                model.train()
                batch_count = 0
                if config.batch_data_while_training:
                        print(f'\n>>> epoch: {epoch+1}')
                for batch in dataset['train_dataset']:
                        input = batch['input'].to(rank) # shape: (batch, seq)
                        label = batch['label'].to(rank) # shape: (batch, seq)

                        out, _ = model(input) # (batch, seq, vocab_size)

                        # loss
                        loss = loss_function(out.contiguous().view(-1, out.size(-1)), torch.eye(config.vocab_size)[label.view(-1)])
                        loss.backward()
                        
                        # optimization
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        batch_count+=1
                        if config.batch_data_while_training:
                                print(f'>>> batch: {batch_count}, loss: {loss.item():6.3f}')
                        # break

                # to save the model instance at end of every epoch
                file = get_weights_file_path(config,f'{epoch:03d}')
                dist.barrier()
                torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, f = file)

                if not config.batch_data_while_training:
                        print(f'\n>>> epoch: {epoch+1}, loss: {loss.item():6.3f}')

                # validation step
                if config.validation_step_while_training:
                        if epoch % config.validation_step_frequency==0:
                                print('>> Validation step..')
                                generate_text(model, config, rank, text=dataset['val_dataset']['text'], unconditional=True)
                scheduler.step()

        optimizer.zero_grad()

        end.record()
        torch.cuda.synchronize()
        print(f'\nTraining complete with total time {start.elapsed_time(end)/1000:.3f} sec.')

        cleanup()

def init_fsdp(world_size, model, config):
        assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"

        mp.spawn(fsdp_train,
                 args=(world_size, model, config),
                 nprocs=world_size,
                 join=True)