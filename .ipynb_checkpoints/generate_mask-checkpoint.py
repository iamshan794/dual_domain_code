import numpy as np 
from sampling import *
import os
import argparse 

def generate_mask(pattern="gaussian2d",image_shape=(128,128),factor=.20,mask_path='datadrive/predata/masks',mode='Train'):
    mask_type={'uniform1d':uniform1d,'uniform2d':uniform2d,'centered_circle':centered_circle,'centered_lines':centered_lines,'uniform_pattern':uniform_pattern,'gaussian1d':gaussian1d,'gaussian2d':gaussian2d,'poisson_disc2d':poisson_disc2d,'poisson_disc_pattern':poisson_disc_pattern}
    mask_fn=mask_type[pattern]
    u_m=mask_fn(image_shape,factor)
    fname=mode+'_'+pattern+'_'+str(factor)
    np.save(os.path.join('/home/mainuser',mask_path,fname+'.npy'),u_m[np.newaxis,:,:],allow_pickle=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_pattern',type=str, default='gaussian2d') 
    parser.add_argument('--mode', type=str,default='Train') 
    config=parser.parse_args()
    
    pat=config.sampling_pattern 
    md=config.mode
    generate_mask(pattern=pat,mode=md)