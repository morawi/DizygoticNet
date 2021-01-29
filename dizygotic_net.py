# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:14:31 2021

@author: malrawi



Details about the models are below:
https://github.com/lukemelas/EfficientNet-PyTorch


Name	          #Params	Top-1-Acc.	Pretrained
----------------------------------------------------
efficientnet-b0	    5.3M	 76.3	        ✓
efficientnet-b1	    7.8M	 78.8	        ✓
efficientnet-b2	    9.2M	 79.8	        ✓
efficientnet-b3	    12M	     81.1	        ✓
efficientnet-b4	    19M	     82.6	        ✓
efficientnet-b5	    30M	     83.3	        ✓
efficientnet-b6	    43M	     84.0	        ✓
efficientnet-b7	    66M	     84.4	        ✓
----------------------------------------------------

There is also a new, large efficientnet-b8 pretrained model that is only available in advprop form. When using these models, replace ImageNet preprocessing code as follows:

if advprop:  # for models using advprop pretrained weights
    normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])




"""

import torch.nn as nn
from efficientnet_pytorch import EfficientNet


''' 


'''

class DizygoticNet(nn.Module):

    def __init__(self, num_classes, 
                 model_name1='efficientnet-b0', 
                 model_name2='efficientnet-b0',
                 pre_trained1 = True,
                 pre_trained2= True):
        super().__init__()

        # EfficientNet    
        if pre_trained1:
            self.network1 = EfficientNet.from_pretrained(model_name1, 
                             num_classes= num_classes, include_top=True) # in_channels=1)
            # model = EfficientNet.from_pretrained("efficientnet-b0", advprop=True)
        else:
            self.network1 = EfficientNet.from_name(model_name1, 
                             num_classes= num_classes, include_top=True) # in_channels=1)
            
        if pre_trained2:
            self.network2 = EfficientNet.from_pretrained(model_name2, 
                             num_classes= num_classes, include_top=True) # in_channels=1)
        else:
            self.network2 = EfficientNet.from_name(model_name2, 
                             num_classes= num_classes, include_top=True) # in_channels=1)
     
    
    def forward(self, x1, x2=None):
        out1 = self.network1(x1)
        out2 = self.network2(x2) if x2 is not None else 0
        
        return out1+out2
    
    # A model with the final layers






# https://github.com/lukemelas/EfficientNet-PyTorch/pull/208
# model = EfficientNet.from_name("efficientnet-b0", num_classes=2, include_top=True, in_channels=1)
# #self.network1 = EfficientNet.from_pretrained(model_name)        
        # # Replace last layer
        # self.network1._fc = nn.Sequential(nn.Linear(self.network1._fc.in_features, 512), 
        #                                  nn.ReLU(),  
        #                                  nn.Dropout(0.25),
        #                                  nn.Linear(512, 128), 
        #                                  nn.ReLU(),  
        #                                  nn.Dropout(0.50), 
        #                                  nn.Linear(128, num_classes))    
    