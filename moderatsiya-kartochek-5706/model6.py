
from peft import LoraConfig, get_peft_model
from torch import nn
import torch
from torch.nn import Parameter
from modules.blippa import Blippa
from modules.augment import RandomErase
from p_config import blip_config
import numpy as np



class Model(nn.Module):

    def __init__(self,pretrained:str = None,train:bool =True):
        super().__init__()

        self.blip = Blippa(blip_config)
        
        
        self.q = nn.Linear(768,768)
        self.k = nn.Linear(768,768)
        self.v = nn.Linear(768,577)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.act = nn.GELU()
        self.blip.requires_grad_(False)

        if pretrained:
            self.load_state_dict_(torch.load(pretrained),ignore_broken = True)
        
        
    def save(self,dir,infer = False):
            for name, xx in self.named_parameters():
                if xx.requires_grad or infer:
                        torch.save(xx,f"{dir}/{name}.pth")

    def load(self,weights = 'weights'):
        for name, xx in self.named_parameters():
            if xx.requires_grad:
                if isinstance(xx, Parameter):
                    xx = xx.data
                try:
                    if not torch.cuda.is_available():
                        xx.copy_(torch.load(f'{weights}/{name}.pth',map_location=torch.device('cpu')))
                        print(f"{name} loaded")
                    else:
                        xx.copy_(torch.load(f'{weights}/{name}.pth'))
                        print(f"{name} loaded")
                except FileNotFoundError:
                    print(f"{name}.pth not found")
                except RuntimeError:
                    print(f"{name}.pth not match")

                    
    def load_state_dict_(self, state_dict,ignore_broken = False):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if ignore_broken:
                try:
                    own_state[name].copy_(param)
                except RuntimeError:
                    continue
            else:
                own_state[name].copy_(param)
        
        

    def forward(self,inpup):
            b,e = self.blip(pixel_values = inpup)
            if self.training:
                b = b.float()
                e = e.float()
            b = self.dropout1(b)
            e = self.dropout2(e)
            attention = torch.tanh((self.q(b))@(self.k(e).permute(0,2,1)))@(self.v(b)).permute(0,2,1)
            return attention.squeeze()
    

if __name__ == '__main__':
    model = Model().cuda()
