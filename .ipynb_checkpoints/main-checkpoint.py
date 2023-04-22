import os
import argparse
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import torch.optim as optim
import math
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from dataset import TrainDataSet, ValidDataSet, TestDataSet
from Ktransformer import Transformer
from train import train
import csv
from torchviz import make_dot 
from torch.utils.tensorboard import SummaryWriter
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
def str2bool(v):
    return v.lower() in ('true')
def save_param_list(model,fname):
    
    file = open(fname, 'w+', newline ='')
    with file:
        write=csv.writer(file)
        for n, p in model.named_parameters():
            #print('Parameter name:', n)
            write.writerow([n])
            #print(p.data)
            #print('requires_grad:', p.requires_grad)
    

def load_freezable_params(fname):
    fparam=[]
    with open(fname) as file_obj:

        reader_obj = csv.reader(file_obj)
       
        for row in reader_obj:
            fparam.append(row)
    fparam=[fparam[i][-1] for i in range(len(fparam)) if len(fparam[i])>0 ]
    return fparam

def load_and_fparams(model, model_Path, device):
   
    # Load the best parameters
    checkpoint = torch.load(model_Path)
    itrlen=len(checkpoint['model_state_dict'])
    checkpoint_new=checkpoint['model_state_dict']
    #print(checkpoint['model_state_dict'].keys())
    #checkpoint_new={}
    #for idx,itr in enumerate(checkpoint['model_state_dict'].keys()):
        #checkpoint_new[itr[7:]]=checkpoint['model_state_dict'][itr]
    
    model.load_state_dict(checkpoint_new)
    model.to(device)
    #save_param_list(model,'params.csv')
    #EDIT params.csv TO MAKE LIST OF FREEZABLE PRAMS
    fparam=load_freezable_params('params.csv')
    print(fparam)
    #print(fparam)
    for n, p in model.named_parameters():
            
            if n in fparam:
                p.requires_grad=False
                
    print("FROZEN PARAMS ARE :")
    
    for n, p in model.named_parameters():
            
            if p.requires_grad==False:
                print(n)
                
    return model
                
                
                
def set_seed(config):
  seed = config.seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  cudnn.benchmark = True
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  cudnn.benchmark = False
  cudnn.deterministic = True

# -------------------------------------------------------- 读取超参数
parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str,
                    help='File path to log each training, including best model, visualization, tensorboard log file ...') # ！
parser.add_argument('--checkpoint', default=None, help='Path to checkpoint')
parser.add_argument('--resume_train', type=str2bool, default='False',
                    help='Resume learning rate, epoch num of an interrupted training')
parser.add_argument('--gpu', type=str, default='0,1,2,3')

# ----------------- Learning Rate, Loss and Regularizations

parser.add_argument('--epoch_num', type=int, default=200//2)

parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--kspace_loss', type=str2bool, default='True', help='MSE loss in k-space')
parser.add_argument('--img_loss', type=str2bool, default='True', help='MSE loss in image domain')

parser.add_argument('--lr', type=float, default=5e-4, help='Maximum learning rate')

parser.add_argument('--lr_weights', nargs='+', type=float,
                    help='Loss weights for each LR decoder layers repectively(must align with num_LRcoder_layers)',
                    default=[0.3, 0.3, 0.3, 0.3])
parser.add_argument('--hr_weights', nargs='+', type=float,
                    help='Loss weights for each HR decoder layers repectively(must align with num_HRdecoder_layers)',
                    default=[0.3, 0.3, 0.3, 0.3, 0.3, 1.0])
parser.add_argument('--conv_weight', type=float, default=1.0,
                    help='Loss weight for image refinement module')
#####################here####################################
parser.add_argument('--pure_LR_training_epoch', nargs='+', type=int, default=2//2,
                    help='Training epoch for LR decoder only')
parser.add_argument('--pure_K_training_epoch', nargs='+', type=int, default=4//2,
                    help='Training epoch for k-space LR+HR decoder before image RM, must be large than pure_LR_training_epoch')

parser.add_argument('--l2norm', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--reassign_mask', type=int, default=1, help='Reassign undersampling masks to each sample every N epoch')

# ----------------- Model Structure

parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--num_encoder_layers', type=int, default=3)
parser.add_argument('--num_LRdecoder_layers', type=int, default=3)
parser.add_argument('--num_HRdecoder_layers', type=int, default=3)
parser.add_argument('--dim_feedforward', type=int, default=1024)

parser.add_argument('--hr_conv_channel', type=int, default=64)
parser.add_argument('--hr_conv_num', type=int, default=3)
parser.add_argument('--hr_kernel_size', type=int, default=3)

# ----------------- Dataset

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--valid_batch_size', type=int, default=16)

parser.add_argument('--lr_size', type=int, default=64, help='Resolution of LR decoder reconstruction')

parser.add_argument('--train_hr_data_path', type=str, help='Path to the k-space data', default='xxx/xxx.npy')
parser.add_argument('--train_lr_data_path', type=str, help='Path to the downsampled k-space data', default='xxx/xxx.npy')
parser.add_argument('--train_mask_path', type=str, help='Path to the undersampling masks', default='xxx/xxx.npy')

parser.add_argument('--valid_hr_data_path', type=str)
parser.add_argument('--valid_lr_data_path', type=str)
parser.add_argument('--valid_mask_path', type=str)

parser.add_argument('--performTL', type=str, default="False")
parser.add_argument('--get_params', type=str, default="False")
parser.add_argument('--visualize_model', type=str, default="False")
parser.add_argument('--modelPath', type=str, help="PRETRAINED MODEL PATH",default="/home/mainuser/datadrive/models/OAS G_2D_0.4_center16.pth")

config = parser.parse_args()

set_seed(config)

train_path = config.train_hr_data_path
train_lr_path = config.train_lr_data_path
valid_path = config.valid_hr_data_path
valid_lr_path = config.valid_lr_data_path
train_mask_path = config.train_mask_path
valid_mask_path = config.valid_mask_path

def data_loader():
    print('Start Loading Dataset from %s, \nMask from %s' % (config.train_hr_data_path, config.train_mask_path))
    t1 = time.time()
    trainSet = TrainDataSet(train_path, train_lr_path, train_mask_path)
    trainLoader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=2)
    print('Train data num : %d' % len(trainSet))
    # val和valid用的是同一个mask
    validSet = ValidDataSet(valid_path, valid_lr_path, valid_mask_path)
    validLoader = DataLoader(validSet, batch_size=config.valid_batch_size, shuffle=True, num_workers=1)
    print('valid data num : %d ' % len(validSet))
    print('Sampled Ratio: %.4f ' % (trainSet.sampled_num/trainSet.unsampled_num))
    print('Dataset load time : %d \n' % (time.time() - t1))
    return trainSet, trainLoader, validSet, validLoader

trainSet, trainLoader, validSet, validLoader = data_loader()

model = Transformer(lr_size=config.lr_size,
                    d_model=config.d_model,
                    # Multi Head
                    nhead=config.n_head,
                    # Layer Number
                    num_LRdecoder_layers=config.num_LRdecoder_layers,
                    num_HRdecoder_layers=config.num_HRdecoder_layers,
                    num_encoder_layers=config.num_encoder_layers,
                    # MLP in Transformer Block
                    dim_feedforward=config.dim_feedforward,
                    # Refine Module CNN
                    HR_conv_channel=config.hr_conv_channel,
                    HR_conv_num=config.hr_conv_num,
                    HR_kernel_size=config.hr_kernel_size,
                    dropout=config.dropout,
                    activation="relu")

def check_gpu_info():
  
    
  if not torch.cuda.is_available():
    print("No GPU found")
    return
  

  device_count = torch.cuda.device_count()
  for i in range(device_count):
    device = torch.device(f"cuda:{i}")
    print(f"Device {i}: {torch.cuda.get_device_name(device)}")
    print("ID",torch.cuda.current_device())
    print(f"\tDevice ID: {device.index}")
    gpu_props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {gpu_props.name}")
    print(f"\tTotal memory: {gpu_props.total_memory / 1024**3:.2f} GB")
    print(f"\tMultiprocessors: {gpu_props.multi_processor_count}")

    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = torch.cuda.memory_reserved(device) - allocated_memory
    print(f"\tTotal memory: {total_memory/1e9:.2f} GB")
    print(f"\tAllocated memory: {allocated_memory/1e9:.2f} GB")
    print(f"\tFree memory: {free_memory/1e9:.2f} GB")
    


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    
print("GPU INFO",config.gpu)
check_gpu_info()
device = torch.device('cuda') 
model = nn.DataParallel(model)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.l2norm)

def lambda_rule(epoch):
    return 0.5 * (1 + math.cos(math.pi * (epoch) / (config.epoch_num)))

lr_sch = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)  # cosine annealing

start_epoch = 1
best_val_psnr = best_val_ssim = 0.0
stage = 'LR'

if config.resume_train:
  checkpoint = torch.load(config.checkpoint)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  lr_sch.load_state_dict(checkpoint['lr_sch_state_dict'])
  best_val_psnr = checkpoint['best_val_psnr']
  best_val_ssim = checkpoint['best_val_ssim']
  stage = checkpoint['stage']
  start_epoch = int(checkpoint['epoch']) + 1
  print('Resume Training from %s, Epoch %d, Stage %s, Best_PSNR:%.2f' % (config.checkpoint, start_epoch, stage, best_val_psnr))
elif config.checkpoint:
  checkpoint = torch.load(config.checkpoint)
  model.load_state_dict(checkpoint['model_state_dict'])
  print('Load checkpoint from %s, Best_PSNR:.3f' % (config.checkpoint, checkpoint['best_val_psnr']))

print('Output file locate at : %s' % os.path.join('Log', config.output_dir))

if config.performTL=="True":
    #save_param_list(model,'params.csv')
    model=load_and_fparams(model,config.modelPath,device)
    
if config.get_params=="False" and config.visualize_model=="False":
    train(model=model,
          optimizer=optimizer,
          lr_sch=lr_sch,
          start_epoch=start_epoch,
          best_valid_psnr=best_val_psnr,
          best_valid_ssim=best_val_ssim,
          device=device,
          config=config,
          trainSet=trainSet, trainLoader=trainLoader,
          validSet=validSet, validLoader=validLoader)
else:
    
    save_param_list(model,'params.csv')
    print("LIST OF PARAMETERS STORED")

if config.visualize_model=="True":
    print(model)
    '''
    trainSet.reassign_mask()
    batch=next(iter(trainLoader))
    sampled_k = batch['sampled_k'].to(device)  # [bs, input_len, c]
    sampled_pos_norm = batch['sampled_pos_norm'].to(device)  # [bs, input_len, 2]

    unsampled_pos_norm = batch['unsampled_pos_norm'].to(device)  # [bs, query_len, 2]
    unsampled_pos = batch['unsampled_pos']                       # [bs, query_len, 2]

    k_us = batch['k_us'].to(device)  # [bs, h, w, 2]
    i_gt = batch['i_gt'].to(device)  # [bs, h, w, 2]
    k_gt = batch['k_gt'].to(device)
    mask = batch['selected_mask'].to(device)

    LR_i_gt = batch['LR_i_gt'].to(device)  # [bs, h, w, 2]
    LR_k_gt = batch['LR_k_gt'].to(device)
    LR_pos_norm = batch['LR_pos_norm'].to(device)
    checkpoint = torch.load(config.modelPath)
    checkpoint_new=checkpoint['model_state_dict']
    
    model.load_state_dict(checkpoint_new)
    conv_weight=checkpoint['conv_weight']
    checkpoint_new=dict(zip(checkpoint_new.keys(),checkpoint_new.values()))
    model.eval() 
    print(checkpoint['conv_weight']) #ynet=model(src=sampled_k,lr_pos=LR_pos_norm,src_pos=sampled_pos_norm,hr_pos=unsampled_pos_norm,k_us=k_us,unsampled_pos=unsampled_pos,up_scale=2,mask=mask,conv_weight=conv_weight,stage='LR')
    #ynet=[torch.tensor(k) for k in ynet]
    #print([type(k) for k in ynet])
    #make_dot(ynet,params=dict(list(model.named_parameters())))
    writer = SummaryWriter("torchlogs/")
    writer.add_graph(model,(sampled_k, LR_pos_norm, sampled_pos_norm, unsampled_pos_norm, k_us, unsampled_pos,torch.Tensor(2), mask, torch.Tensor(1)))
    writer.close()
'''

