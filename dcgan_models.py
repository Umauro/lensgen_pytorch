"""
    Discriminator and Generator models classes
"""

import torch 
import torch.nn as nn

NC = 1
NH = 101
NW = 101
ALPHA = 0.2

class Discriminator(nn.Module):
    def __init__(self):
        """
            Define the network architecture
            using a Sequential container
        """
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            # Stem Group 
            nn.Conv2d(NC,32,3,stride=1,padding=1),
            nn.LeakyReLU(ALPHA,inplace=True),
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
            nn.LeakyReLU(ALPHA,inplace=True),
            # Task Group
            nn.Flatten(),
            nn.Dropout(0.4,inplace=True),
            # Flatten result from 13x13x256
            nn.Linear(13*13*256,1),
            nn.Sigmoid()
        )
    
    def forward(self,input):
        return self.main(input)

