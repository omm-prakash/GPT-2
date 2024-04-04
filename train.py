import torch
import argparse
import torch.nn as nn
from gpt2.utils import *
from gpt2.model import get_gpt2
from gpt2.sample import generate_text

from dataset import get_dataset
from fast import init_dpp, init_fsdp

def parser():
        parser = argparse.ArgumentParser(description='Training GPT-2.')
        parser.add_argument("--config", type=str, required=False, default='config.yml')
        parser.add_argument("--load_pretrained", default=False, action='store_true')
        parser.add_argument("--data_path", type=str, required=False, default='data/data.txt')
        parser.add_argument("--fsdp", required=False, default=False, action='store_true')
        parser.add_argument("--dpp", required=False, default=False, action='store_true')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

        args = parser.parse_args()
        return args

def train(model, config, device):
        print('\nLoading dateset..')
        dataset = get_dataset(config)
        
        print('\nHardware setup..')
        device_name = torch.cuda.get_device_name(device.index) if device.type == 'cuda' else 'CPU'
        print('>> Detected device:', device_name)

        # changing model configuration if multiple GPU avialable
        if torch.cuda.device_count() > 1:
                print(">> Using", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)
        loss_function = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing).to(device) 
        
        model_file = latest_weights_file_path(config) if config.preload=='latest' else get_weights_file_path(config, config.preload) if config.preload else None        
        init_epoch = 0
        print(f'\nModel Training on {device}..')
        if model_file:
                state = torch.load(model_file)
                init_epoch = state['epoch']+1
                print(f'>> Resuming model training from epoch no. {init_epoch}')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
        else:
                print('>> No model to preload, starting from scratch.')

        # training loop
        if torch.cuda.is_available(): 
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
        else:
                tm = Time()
                tm.start(message='training model')
        for epoch in range(init_epoch, config.epochs):
                torch.cuda.empty_cache()
                model.train()
                batch_count = 0
                if config.batch_data_while_training:
                        print(f'\n>>> epoch: {epoch+1}')
                for batch in dataset['train_dataset']:
                        input = batch['input'].to(device) # shape: (batch, seq)
                        label = batch['label'].to(device) # shape: (batch, seq)

                        if torch.cuda.device_count() > 1:
                                out, _ = model.module(input) # (batch, seq, vocab_size)
                        else:
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
                                generate_text(model, config, device, text=dataset['val_dataset']['text'], unconditional=True)
        print()
        if torch.cuda.is_available():
                generate_text(model, )
                end.record()
                torch.cuda.synchronize()
                print(f'\nTraining complete with total time {start.elapsed_time(end)/1000:.3f} sec.')
        else:
                tm.end()
                print(f'\nTraining complete.')
        return 

if __name__ == '__main__':
        args = parser()
        config = load_config(args.config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(args.seed)
        world_size = torch.cuda.device_count()

        # load model from sctrach or load openai gpt2 pretrained model
        if args.load_pretrained:
                model = load_model_states(device, config, pretrained=args.load_pretrained)
        else:
                model = get_gpt2(config)

        # training strategy
        if args.fsdp:
                init_fsdp(world_size, model, config)
        elif args.dpp:
                init_dpp(world_size, model, config)
        else:
                train(model, config, device)
