from torchvision import transforms as T
from PIL import Image
import os
import torch
def init_model(model,weights,device):
    return model(weights,train=False).to(device)

denormalize = T.Compose([T.Normalize(mean = [ 0., 0., 0. ],
                                    std = [ 1/0.229, 1/0.224, 1/0.225 ]),
            T.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                    std = [ 1., 1., 1. ]),
            ])


