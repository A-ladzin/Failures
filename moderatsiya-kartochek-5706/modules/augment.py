import numpy as np
from torchvision import transforms as T
from PIL import Image

class RandomErase:
    def __init__(self,p=1,n = (1,50), scale = (0.001,0.5),ratio = (0.01,100),temperature = 0.005,colored = True):
        super().__init__()
        scales = np.linspace(scale[0],scale[1],n[1]+1)
        probs = (scale[1]-scales)**(1/temperature)
        probs = probs/ probs.sum()
        self.n= n
        self.probs = probs
        self.scales = scales
        self.ratio = ratio
        self.p = p
        self.colored = colored
        

    def get_transform(self,input):
        if self.colored:
            value = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
        else:
            value = 0

        return T.RandomErasing(p = 1, scale = (c:=np.random.choice(self.scales,p=self.probs),c),ratio=self.ratio,value = value)(input)


    def __call__(self,input):
        if np.random.rand() > self.p:
            return input
        for i in range(np.random.randint(self.n[0],self.n[1])):
            input = self.get_transform(input)
        return input
    


class ResizeWithPadding:
    def __init__(self, target_size, fill=0):
        self.target_size = target_size
        self.fill = fill
        self.tt = T.ToTensor()

    def __call__(self, img):
        # Get original dimensions
        original_width, original_height = img.size

        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height
        target_aspect_ratio = self.target_size[0] / self.target_size[1]

        if aspect_ratio > target_aspect_ratio:
            # Image is wider than the target aspect ratio
            new_width = self.target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            # Image is taller than the target aspect ratio
            new_height = self.target_size[1]
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.BILINEAR)

        # Calculate padding
        padding_left = (self.target_size[0] - new_width) // 2
        padding_top = (self.target_size[1] - new_height) // 2
        padding_right = self.target_size[0] - new_width - padding_left
        padding_bottom = self.target_size[1] - new_height - padding_top

        # Pad the image
        padding = (padding_left, padding_top, padding_right, padding_bottom)
        padded_img = T.functional.pad(resized_img, padding, fill=self.fill)

        return self.tt(padded_img)
