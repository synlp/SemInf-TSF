import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if len(optimizer.param_groups) > 1:
        if args.lradj == 'type1':
            main_lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
            projection_lr = getattr(args, 'projection_learning_rate', args.learning_rate * 0.5) * (0.5 ** ((epoch - 1) // 1))
            llm_lr = getattr(args, 'llm_learning_rate', args.learning_rate * 0.1) * (0.5 ** ((epoch - 1) // 1))
            
        elif args.lradj == 'type2':
            lr_table = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
            if epoch in lr_table:
                main_lr = lr_table[epoch]
                projection_lr = main_lr * 0.5 
                llm_lr = main_lr * 0.1   
            else:
                return  
                
        elif args.lradj == 'type3':
            if epoch < 3:
                main_lr = args.learning_rate
                projection_lr = getattr(args, 'projection_learning_rate', args.learning_rate * 0.5)
                llm_lr = getattr(args, 'llm_learning_rate', args.learning_rate * 0.1)
            else:
                decay_factor = 0.9 ** ((epoch - 3) // 1)
                main_lr = args.learning_rate * decay_factor
                projection_lr = getattr(args, 'projection_learning_rate', args.learning_rate * 0.5) * decay_factor
                llm_lr = getattr(args, 'llm_learning_rate', args.learning_rate * 0.1) * decay_factor
                
        elif args.lradj == 'constant':
            main_lr = args.learning_rate
            projection_lr = getattr(args, 'projection_learning_rate', args.learning_rate * 0.5)
            llm_lr = getattr(args, 'llm_learning_rate', args.learning_rate * 0.1)
            
        elif args.lradj in ['3', '4', '5', '6']:
            if args.lradj == '3':
                decay_epoch = 10
            elif args.lradj == '4':
                decay_epoch = 15
            elif args.lradj == '5':
                decay_epoch = 25
            elif args.lradj == '6':
                decay_epoch = 5
                
            if epoch < decay_epoch:
                main_lr = args.learning_rate
                projection_lr = getattr(args, 'projection_learning_rate', args.learning_rate * 0.5)
                llm_lr = getattr(args, 'llm_learning_rate', args.learning_rate * 0.1)
            else:
                main_lr = args.learning_rate * 0.1
                projection_lr = getattr(args, 'projection_learning_rate', args.learning_rate * 0.5) * 0.1
                llm_lr = getattr(args, 'llm_learning_rate', args.learning_rate * 0.1) * 0.1
                
        elif args.lradj == 'TST':
            lrs = scheduler.get_last_lr()
            if len(lrs) >= 3:
                main_lr = lrs[0]
                projection_lr = lrs[1]
                llm_lr = lrs[2]
            else:
                main_lr = lrs[0]
                projection_lr = main_lr * 0.5
                llm_lr = main_lr * 0.1
        else:
            main_lr = args.learning_rate
            projection_lr = getattr(args, 'projection_learning_rate', args.learning_rate * 0.5)
            llm_lr = getattr(args, 'llm_learning_rate', args.learning_rate * 0.1)

        for param_group in optimizer.param_groups:
            if param_group.get('name') == 'llm':
                param_group['lr'] = llm_lr
            elif param_group.get('name') == 'projection':
                param_group['lr'] = projection_lr
            else:
                param_group['lr'] = main_lr
        
        if printout:
            print('Updating learning rates:')
            for param_group in optimizer.param_groups:
                print(f'  {param_group.get("name", "default")}: {param_group["lr"]:.6f}')
    
    else:
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        elif args.lradj == 'type3':
            lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif args.lradj == 'constant':
            lr_adjust = {epoch: args.learning_rate}
        elif args.lradj == '3':
            lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
        elif args.lradj == '4':
            lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
        elif args.lradj == '5':
            lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
        elif args.lradj == '6':
            lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
        elif args.lradj == 'TST':
            lr_adjust = {epoch: scheduler.get_last_lr()[0]}
        
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        if hasattr(model.model, 'backbone') and hasattr(model.model.backbone, 'large_model_integration'):
            lmi = model.model.backbone.large_model_integration
            if hasattr(lmi, 'use_lora') and lmi.use_lora:
                lmi.large_model.save_pretrained(os.path.join(path, 'lora_adapter'))
            else:
                torch.save(lmi.large_model.state_dict(), os.path.join(path, 'large_model.pth'))
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))