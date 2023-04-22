import torch 
import torch.nn as nn

class DualDomainLoss(nn.Module):
    
    def __init__(self,device,lambdal=0):
        super(DualDomainLoss, self).__init__()
        self.image_loss=nn.MSELoss()
        self.kspace_loss=nn.MSELoss() 
        self.to(device)
        self.lambda_l=lambdal
    def forward(self,img,img_gt,k,k_gt):
        #print("MSELOSS",img.shape,img_gt.shape,k.shape,k_gt.shape)
        return self.lambda_l*self.image_loss(img,img_gt)+self.kspace_loss(k,k_gt)