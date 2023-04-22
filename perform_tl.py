import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchviz import make_dot 
import argparse

from dataset import TestDataSet
from Ktransformer import Transformer
from utils import *
from fftc import *
import csv

os.environ['TORCH_USE_CUDA_DSA'] = '1'
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

def perform_tl(model, model_Path, device,config):
   
    # Load the best parameters
  
    print("USING DEVICE",device)
    checkpoint = torch.load(model_Path)
    itrlen=len(checkpoint['model_state_dict'])
    checkpoint_new={}
    for idx,itr in enumerate(checkpoint['model_state_dict'].keys()):
        checkpoint_new[itr[7:]]=checkpoint['model_state_dict'][itr]
    print(checkpoint.keys())
    model.load_state_dict(checkpoint_new)
    print(checkpoint['optimizer_state_dict'].keys())
    model.to(device)
    #save_param_list(model,'params.csv')
    #EDIT params.csv TO MAKE LIST OF FREEZABLE PRAMS
    fparam=load_freezable_params('params.csv')
    
    for n, p in model.named_parameters():
            
            if n in fparam:
                p.requires_grad=False
    print("FROZEN PARAMS ARE: ")
    for n, p in model.named_parameters():
            if p.requires_grad==False:
                print(n)
                
    conv_weight=checkpoint['conv_weight']
    state=checkpoint['optimizer_state_dict']['state']
    visualize_model(model,config,conv_weight,state,device)
    
    
def visualize_model(model,config,conv_weights,state,device):
    """

    Args:
      src: [bs, src_len, c] intensity of sampled points
      lr_pos: [bs, lh*lw, 2] normalized coordinates of LR query points
      src_pos: [bs, src_len, 2] normalized coordinates of sampled points
      hr_pos: [bs, query_len, 2] normalized coordinates of unsampled points
      k_us: [bs, h, w, c] zero-filled specturm
      mask: [bs, h, w, c] undersampling mask, 1 means unsampled and 0 means sampled
      unsampled_pos: [bs, query_len, 2] coordinates of unsampled points(unnormalized)
      up_scale: LR upsample ratio to HR

   """

    h,w,c=(128,128,1)
    bs=1
    lh,lw=128//2,128//2
    src_len=128*128
    query_len=64*64
    src=torch.zeros([bs, src_len, c]).to(device)
    lr_pos=torch.zeros([bs, lh*lw, 2]).to(device)
    src_pos=torch.zeros([bs, src_len, 2]).to(device)
    hr_pos=torch.zeros([bs, query_len, 2]).to(device)
    k_us=torch.zeros([bs, h, w, c]).to(device)
    mask=torch.zeros([bs, h, w, c]).to(device)
    unsampled_pos=torch.zeros([bs, query_len, 2]).to(device)
    up_scale=2
    stage='LR'

    
    model.eval()
    make_dot(model(src,lr_pos,src_pos,hr_pos,k_us,unsampled_pos,up_scale,mask,conv_weight=conv_weights,stage='LR'),params=dict(list(model.named_parameters())))
    
if __name__ == '__main__':
  # -------------------------------------------------------- 读取超参数

  parser = argparse.ArgumentParser()

  # ----------------- Testing ID

  parser.add_argument('--output_dir', type=str, default=None, help="path to record the evaluation results")

  parser.add_argument('--gpu', type=str, default='0,1,2,3')

  # ----------------- Model Structure

  parser.add_argument('--modelPath', type=str, help="checkpoint to evaluate",default="/home/mainuser/datadrive/models/OAS G_2D_0.4_center16.pth")

  parser.add_argument('--d_model', type=int, default=256)
  parser.add_argument('--n_head', type=int, default=4)
  parser.add_argument('--num_encoder_layers', type=int, default=4)
  parser.add_argument('--num_LRdecoder_layers', type=int, default=4)
  parser.add_argument('--num_HRdecoder_layers', type=int, default=6)
  parser.add_argument('--dim_feedforward', type=int, default=1024)

  parser.add_argument('--dropout', type=float, default=0.0)

  parser.add_argument('--HR_conv_channel', type=int, default=64)
  parser.add_argument('--HR_conv_num', type=int, default=3)
  parser.add_argument('--HR_kernel_size', type=int, default=3)
  parser.add_argument('--conv_weight', type=float, default=1.0)

  # ----------------- Dataset Control

  parser.add_argument('--batch_size', type=int, default=24)
  parser.add_argument('--data_path', type=str, help='Path to the k-space data', default='/home/mainuser/datadrive/predata/data/valid/hr/numpy_valid_hr128.npy')
  parser.add_argument('--mask_path', type=str, help='Path to the undersampling masks', default='/home/mainuser/datadrive/predata/masks/Train_gaussian2d_0.2.npy' )

  config = parser.parse_args()

  # Prepare Dataset
  testPath = config.data_path
  test_maskPath = config.mask_path

  testSet = TestDataSet(testPath, test_maskPath)
  testLoader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, num_workers=8)

  # Prepare Model
  model = Transformer(lr_size=64,
                      d_model=config.d_model,
                      # Multi Head
                      nhead=config.n_head,
                      # Layer Number
                      num_LRdecoder_layers=config.num_LRdecoder_layers,
                      num_HRdecoder_layers=config.num_HRdecoder_layers,
                      num_encoder_layers=config.num_encoder_layers,
                      # MLP in Transformer Block
                      dim_feedforward=config.dim_feedforward,
                      # HR Conv
                      HR_conv_channel=config.HR_conv_channel,
                      HR_conv_num=config.HR_conv_num,
                      HR_kernel_size=config.HR_kernel_size,
                      dropout=config.dropout,
                      activation="relu")

  os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
  device = torch.device('cuda')

  # ___________________________________________________________________________________________
  perform_tl(model, config.modelPath,device,config)
