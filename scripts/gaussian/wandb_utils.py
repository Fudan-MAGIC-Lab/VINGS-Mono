import random
import torch
import os

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

# os.environ["WANDB__SERVICE_WAIT"] = "300"

class Wandber:
    
    def __init__(self, cfg, customize_name):
        self.cfg = cfg
        if cfg['use_wandb']:
            if wandb is None:
                raise ModuleNotFoundError("wandb is required when use_wandb is True.")
            wandb.login(key='bff8fb59d5f957b13c5cba24cd62055f759339d4')
            self.wandb = wandb.init(project="Droid2DAcc", name=customize_name)
            wandb.ensure_configured()
            self.name = customize_name
            self.time_have_started = False
            self.time_have_ended = True
            
    def log_once(self, key, value):
        '''
        operation_type = current, clone, split, prune
        Record the number of "current_num, clone_num, split_num, prune_num"
        这函数应该是加在 scripts/utils/UNI_utils.py 的 Clone(), Split() 里面的，
        Prune() 因为用了两次所以放外面
        '''
        if self.cfg['use_wandb']:
            self.wandb.log({key: value})
    
    def log_time(self, key):
            
        if self.cfg['use_wandb']:
            if not self.time_have_started and self.time_have_ended:
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
                self.start_event.record()
                self.time_have_started = True
                self.time_have_ended = False
            elif self.time_have_started and not self.time_have_ended:
                self.end_event.record()
                torch.cuda.synchronize()
                elapsed_time = self.start_event.elapsed_time(self.end_event)
                self.wandb.log({key: elapsed_time})
                self.time_have_started = False
                self.time_have_ended = True
        
        # else:
        #     if not self.time_have_started and self.time_have_ended:
        #         self.start_event = torch.cuda.Event(enable_timing=True)
        #         self.end_event = torch.cuda.Event(enable_timing=True)
        #         self.start_event.record()
        #         self.time_have_started = True
        #         self.time_have_ended = False
        #     elif self.time_have_started and not self.time_have_ended:
        #         self.end_event.record()
        #         torch.cuda.synchronize()
        #         elapsed_time = self.start_event.elapsed_time(self.end_event)
        #         print('TimePerIter: ', elapsed_time)
        #         self.time_have_started = False
        #         self.time_have_ended = True
            
            
        