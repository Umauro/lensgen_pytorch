"""
    Discriminator and Generator models classes
"""

import torch 
import torch.nn as nn

NC = 1
NH = 101
NW = 101
ALPHA = 0.2

def weights_init(m):
    """
        Weights initializer for both networks

        Glorot uniform for Conv and Linear Layers.
        Keras Default for BatchNorm
    """
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    
    elif class_name.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    
    elif class_name.find('BatchNorm') != -1:
        nn.init.constant_(m.weight.data,1)
        nn.init.constant_(m.bias.data,0)

class Discriminator(nn.Module):
    def __init__(self):
        """
            Define the discriminator network architecture
            using a Sequential container
        """
        super(Discriminator,self).__init__()
        self.stem = nn.Sequential(
            # Stem Group 
            nn.Conv2d(NC,32,3,stride=1,padding=1),
            nn.LeakyReLU(ALPHA,inplace=True)
        )

        self.learner = nn.Sequential(
            # Learner Group
            # First Block
            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(ALPHA,inplace=True),
            # Second Block
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(ALPHA,inplace=True),
            # Third Block
            nn.Conv2d(128,256,3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(ALPHA,inplace=True)
        )

        self.task = nn.Sequential(
            # Task Group
            nn.Flatten(),
            nn.Dropout(0.4),
            # Flatten result from 13x13x256
            nn.Linear(13*13*256,1),
            nn.Sigmoid()
        )
    
    def forward(self,input):
        output = self.stem(input)
        output = self.learner(output)
        output = self.task(output)
        return output

class View(nn.Module):
    """
        View implementation as a module for using it in nn.Sequential
    """
    def __init__(self,shape):
        super(View,self).__init__()
        self.shape = shape
    
    def forward(self,input):
        return input.view(*self.shape)        

class Generator(nn.Module):
    def __init__(self):
        """
            Define the generator network architecture
            using a Sequential container
        """
        super(Generator,self).__init__()
        self.stem = nn.Sequential(
            #STEM GROUP
            nn.Linear(100,128*12*12),
            nn.LeakyReLU(ALPHA,inplace=True),
            View((-1,128,12,12)),
            nn.ZeroPad2d((0,1,0,1))     
        )

        self.learner = nn.Sequential(
            #Conv Group
            #First Block
            nn.ConvTranspose2d(128,128,4,stride=2,padding=0),
            nn.Conv2d(128,128,3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(ALPHA,inplace=True),
            #Second Block
            nn.ConvTranspose2d(128,64,4,stride=2,padding=0),
            nn.Conv2d(64,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(ALPHA,inplace=True),
            #Third Block
            nn.ConvTranspose2d(64,32,4,stride=2,padding=0),
            nn.Conv2d(32,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(ALPHA,inplace=True)
        )

        self.task = nn.Sequential(
            #Task Group
            nn.ConvTranspose2d(32,1,3,stride=1,padding=1),
            nn.Tanh(),
            nn.UpsamplingNearest2d(size=(101,101))
        )
    
    def forward(self,input):
        output = self.stem(input)
        output = self.learner(output)
        output = self.task(output)
        return output

