from layers import *
import math
from fftc import *
from utils import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = grid.astype(np.float32)
  return torch.tensor(grid)



def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = grid.astype(np.float32)
  return torch.tensor(grid)


class Transformer(nn.Module):

  def __init__(self, lr_size, channel=2, d_model=512, nhead=8, num_encoder_layers=6,
               num_LRdecoder_layers=6, num_HRdecoder_layers=6, dim_feedforward=2048,
               HR_conv_channel=64, HR_conv_num=3, HR_kernel_size=5,
               dropout=0.1, activation="relu"):
    super().__init__()

    self.num_HRdecoder_layers = num_HRdecoder_layers

    self.encoder_embed_layer = nn.Sequential(
      nn.Linear(channel, d_model),
      nn.ReLU(inplace=True),
      nn.Linear(d_model, d_model)
    )
    self.encoder_embed_layer_im = nn.Sequential(
      nn.Linear(channel, d_model),
      nn.ReLU(inplace=True),
      nn.Linear(d_model, d_model)
    )

    self.pe_layer = PositionalEncoding(d_model, magnify=250.0)

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

    decoder_layerLR = TransformerDecoderLayerLR(d_model, nhead, dim_feedforward, dropout, activation)
    self.decoderLR = TransformerDecoderLR(d_model=d_model,
                                          lr_size=lr_size,
                                          channel=2,
                                          decoder_layer=decoder_layerLR,
                                          num_layers=num_LRdecoder_layers)

    decoder_layerHR = TransformerDecoderLayerHR(d_model, nhead, dim_feedforward, dropout, activation)
    self.decoderHR = TransformerDecoderHR(d_model=d_model,
                                          k_channel=2,
                                          im_channel=1,
                                          decoder_layer=decoder_layerHR,
                                          num_layers=num_HRdecoder_layers)

    self._reset_parameters()

    self.d_model = d_model
    self.nhead = nhead

  def _reset_parameters(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, src, lr_pos, src_pos, hr_pos,hr_pos_im, k_us, unsampled_pos, up_scale, mask, conv_weight, stage):
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

    Returns:

    """
    #print("FORWARD",src.shape) # 4 , 3276 ,2
    unsampled_pos = unsampled_pos.permute(0, 2, 1).cpu().numpy()  # pos [bs, 2, quey_len]
    
    # encode
    src_embed = self.encoder_embed_layer(src)
    src_pe = self.pe_layer(src_pos)      # [bs, src_len, d]
    Encoder_memory = self.encoder(src_embed, pos=src_pe) # [bs, src_len, d]

    # lr decode
    lr_pe = self.pe_layer(lr_pos)        # [bs, lh*lw, d]
    LR_Trans_outputs, LR_memory = self.decoderLR(Encoder_memory, lr_pe)   # [num_of_layers, img:[bs, lh, lw, c]], [bs, lh*lw, d]

    # upsample in image domain
    LR_i = LR_Trans_outputs[-1].permute(0, 3, 1, 2).contiguous()   # [bs, c, lh, lw]
    Up_LR_i = F.interpolate(LR_i, scale_factor=up_scale, mode='bicubic').permute(0, 2, 3, 1).contiguous() # [bs, h, w, c]
    # back to k space
    Up_LR_k = fft2c_new(Up_LR_i)    # [bs, h, w, c]
    
    if stage == 'LR':
      HR_Trans_outputs = HR_Conv_outputs = [torch.zeros_like(Up_LR_i)] * self.num_HRdecoder_layers
      
        
      return LR_Trans_outputs, Up_LR_i, Up_LR_k, HR_Trans_outputs, HR_Conv_outputs

    else:
      # select unsampled points' predicted values
      unsampled_value_k = []
      unsampled_value_im = []
      for i in range(unsampled_pos.shape[0]):
        unsampled_value_k.append(Up_LR_k[i, :, :, :][unsampled_pos[i, :, :]])  # k [query_len, c]
        #unsampled_value_im.append(Up_LR_i[i, :, :, :][unsampled_pos[i, :, :]])
        

      #print("CHECK AT UNSAMPLED",Up_LR_k[i, :, :, :].shape,unsampled_pos[i, :, :].shape,LR_i[-1,:,:,:].shape)
      #plt.imsave('intermediate.jpg',np.abs(LR_i[-1,1,:,:].detach().cpu().numpy()),cmap='gray')
      unsampled_value_k = torch.stack(unsampled_value_k, dim=0)        # k [bs, query_len, c]
      unsampled_value_im = Up_LR_i
      lr_img=self.encoder_embed_layer_im(LR_Trans_outputs[-1])
      bs,h,w,d_model=lr_img.shape 
      lr_img=lr_img.reshape(bs,h*w,d_model)
      
      # hr decode
      hr_pe_k = self.pe_layer(hr_pos)  # [bs, query_len, d]

      hr_pe_im=self.pe_layer(hr_pos_im) #(bs,h*w,c)->(hr_pe_im)
      
      

     
        
     
      #hr_pe_im=self.pe_layer(torch.arange(128*128)/(128*128))
      HR_Trans_outputs, HR_Conv_outputs = self.decoderHR(lr_memory=LR_memory,
                                                         LR_Trans_outputs=lr_img,
                                                         query_pe_k=hr_pe_k,
                                                         query_pe_im=hr_pe_im,
                                                         query_value_k=unsampled_value_k,
                                                         query_value_im=unsampled_value_im,
                                                         unsampled_pos=unsampled_pos,
                                                         k_us=k_us,
                                                         mask=mask,
                                                         stage=stage)    # [num_of_layers, [bs, h, w, c]]

      return LR_Trans_outputs, Up_LR_i, Up_LR_k, HR_Trans_outputs, HR_Conv_outputs

class TransformerEncoder(nn.Module):

  def __init__(self, encoder_layer, num_layers):
    super().__init__()
    self.layers = _get_clones(encoder_layer, num_layers)
    self.num_layers = num_layers

  def with_pos_embed(self, tensor, pos):
    return tensor if pos is None else tensor + pos

  def forward(self, src, pos):
    output = self.with_pos_embed(src, pos)

    for layer in self.layers:
      output = layer(output)

    return output

class TransformerDecoderLR(nn.Module):

  def __init__(self, d_model, lr_size, channel, decoder_layer, num_layers):
    super().__init__()
    self.layers = _get_clones(decoder_layer, num_layers)
    self.num_layers = num_layers

    self.channel = channel
    self.lr_size = lr_size

    self.LR_predict_layers = []
    self.LR_norm_layers = []
    self.LR_embed_layers = []
    self.ConvBlocks = []

    for i in range(num_layers):
      self.LR_predict_layers.append(nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(inplace=True),
        nn.Linear(d_model, channel),
      ))
      self.LR_norm_layers.append(nn.LayerNorm(d_model, eps=1e-6))
    self.LR_norm_layers = nn.ModuleList(self.LR_norm_layers)
    self.LR_predict_layers = nn.ModuleList(self.LR_predict_layers)

  def forward(self, encoder_memory, lr_pe):

    Transformer_interpredict = []

    input = lr_pe
    for i in range(len(self.layers)):
      output_memory = self.layers[i](input, encoder_memory)   # [bs, lh*lw, d]

      output = self.LR_predict_layers[i](self.LR_norm_layers[i](output_memory))  # k [b, lh*lw, c]
      
      output = torch.reshape(output, (output.shape[0], self.lr_size, self.lr_size, self.channel))   # k [b, lh, lw, c]
      
      output = ifft2c_new(output)   # img [b, lh, lw, c]

      Transformer_interpredict.append(output)
      
      input = output_memory

    return Transformer_interpredict, output_memory

class TransformerDecoderHR(nn.Module):

  def __init__(self, d_model, im_channel,k_channel, decoder_layer, num_layers):
    super().__init__()
    self.im_layers = _get_clones(decoder_layer, num_layers)
    self.k_layers = _get_clones(decoder_layer, num_layers)
    self.num_layers = num_layers

    self.HR_predict_k_layers = []
    self.HR_predict_im_layers=[]
    self.HR_predict_net_layers=[self.HR_predict_k_layers,self.HR_predict_im_layers]
    self.HR_norm_layers = []
    self.HR_embed_layers = []
    self.activation_fn=nn.Sigmoid()
    self.alpha=nn.ParameterList([nn.Parameter(torch.Tensor([0.5]),requires_grad=True) for _ in range(num_layers)])
    self.beta=nn.ParameterList([nn.Parameter(torch.Tensor([0.5]),requires_grad=True) for _ in range(num_layers)])
    self.channel=[im_channel,k_channel]
    for idx,self.HR_predict_layers in enumerate(self.HR_predict_net_layers):
        for i in range(num_layers):
          if idx==1:
              self.HR_predict_k_layers.append(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, self.channel[idx]),
              ))
          else:
              self.HR_predict_im_layers.append(nn.Sequential(
                nn.Conv2d(d_model, d_model,kernel_size=5,padding=5-1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(5,stride=1),
                nn.Conv2d(d_model, self.channel[idx],kernel_size=5,padding=5-1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(5,stride=1)))
          if idx==0:
            self.HR_norm_layers.append(nn.LayerNorm(d_model, eps=1e-6))
            
          #self.ConvBlocks.append(CNN_Block(in_channels=channel, mid_channels=conv_channel, num_convs=conv_num, kernel_size=kernel_size))
    self.HR_norm_layers = nn.ModuleList(self.HR_norm_layers)
    #print(len(self.HR_predict_k_layers),len(self.HR_predict_im_layers))
    self.HR_predict_im_layers,self.HR_predict_k_layers = nn.ModuleList(self.HR_predict_im_layers),nn.ModuleList(self.HR_predict_k_layers)
    self.HR_predict_net_layers=[self.HR_predict_k_layers,self.HR_predict_im_layers]
    
    
    self.HR_embed_k_layer = nn.Sequential(
      nn.Linear(self.channel[1], d_model),
      nn.ReLU(inplace=True),
      nn.Linear(d_model, d_model)
    )
    self.HR_embed_im_layer = nn.Sequential(
                nn.Conv2d(self.channel[0], d_model,kernel_size=5,padding=5-1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(5,stride=1),
                nn.Conv2d(d_model, d_model,kernel_size=5,padding=5-1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(5,stride=1))
    
    
    #self.ConvBlocks = nn.ModuleList(self.ConvBlocks)
  def convert_to_image(self,tsr):
    
    return torch.sqrt(tsr[:,:,:,0]**2 + tsr[:,:,:,1]**2).unsqueeze(-1)

  def forward(self, lr_memory,LR_Trans_outputs,query_pe_k,query_pe_im,query_value_k,query_value_im,unsampled_pos,k_us, mask,stage):
    #lr_memroy is 
    #print("HERE",query_value_im.shape,query_value_k.shape,query_pe.shape)
    #print(self.channel)
    
    k_input = query_pe_k + self.HR_embed_k_layer(query_value_k)
    
    
    query_value_im=self.convert_to_image(query_value_im)
    
    ibs,ih,iw,c=query_value_im.shape

    embedlayerout=self.HR_embed_im_layer(query_value_im.reshape(ibs,c,ih,iw)).reshape(ibs,ih*iw,-1)
   
    bs,_,d_model=k_input.shape

    im_input = query_pe_im + embedlayerout  # value + pe 
    
    im_input=im_input.reshape(bs,d_model,-1).permute(0,2,1)
    
    Transformer_interpredict = []
    CNN_interpredict = []      # [[b, h, w, d]... ...]

    for i in range(len(self.k_layers)):
        
      output_memory_k = self.k_layers[i](k_input, lr_memory)   # [b, query_len, d]
      output_memory_im=self.im_layers[i](im_input,LR_Trans_outputs)  #[b,query_len,d]
      #print("*********************",i,self.HR_norm_layers.__len__())       
      #print(output_memory_im.shape,output_memory_k.shape,"309")
      output_k = self.HR_predict_k_layers[i](self.HR_norm_layers[i](output_memory_k))  # k [b, query_len, c]
      
      output_im = self.HR_predict_im_layers[i](self.HR_norm_layers[i](output_memory_im).reshape(bs,d_model,ih,iw)).reshape(bs,ih,iw,1)

      output_k = fill_in_k(k_us, unsampled_pos, output_k)  # k [bs, h, w, c]
      output_im_coup = ifft2c_new(output_k) # img [bs, h, w, c]
      output_im_coup=torch.sqrt(torch.sum(torch.square(output_im_coup),dim=-1)).unsqueeze(-1)
      output_k_coup= torch.fft.fftshift(torch.fft.fft((output_im)))# [bs,h,w,2] 
      #print(output_im_coup[0,0,0,0],"321")
      output_k_coup=torch.cat((output_k_coup.real,output_k_coup.imag),dim=-1)
    
      #print(output_im.shape,output_im_coup.shape,output_k.shape,output_k_coup.shape,"320")
      assert((output_k_coup.shape==output_k.shape) and (output_im_coup.shape==output_im.shape))
      output_stage_k_net=self.activation_fn(self.alpha[i])*output_k+(1-self.activation_fn(self.alpha[i]))*output_k_coup
      output_stage_im_net=self.activation_fn(self.beta[i])*output_im+(1-self.activation_fn(self.beta[i]))*output_im_coup
    
      Transformer_interpredict.append(output_stage_k_net)
      
      CNN_interpredict.append(torch.cat((output_stage_im_net,torch.zeros_like(output_stage_im_net)),dim=-1))
      
      k_input=output_memory_k
      im_input=output_memory_im
      
      if i+1 == len(self.k_layers):
        #print([param.data for param in self.alpha],[param.data for param in self.beta])
        return Transformer_interpredict, CNN_interpredict

class CNN_Block(nn.Module):
  def __init__(self, in_channels=2, mid_channels=48, num_convs=4, kernel_size=3):
    super(CNN_Block, self).__init__()
    self.convs = []

    # first layers
    self.convs.append(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2))
    self.convs.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
    # N * middle layers
    for i in range(num_convs):
      self.convs.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2))
      self.convs.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
    # final layers
    self.convs.append(nn.Conv2d(mid_channels, in_channels, kernel_size=1))

    self.convs = nn.ModuleList(self.convs)

  def DataConsistency(self, k_rec, k_in, mask):
    # mask中1表示没采样，0表示采样
    k_rec_masked = k_rec * mask
    k_out = k_rec_masked + k_in
    return k_out

  def forward(self, k_sampled, i_in, mask):
    output = i_in.permute(0, 3, 1, 2).contiguous()   # [bs, c, h, w]

    for layer in self.convs:
      output = layer(output)

    output = output.permute(0, 2, 3, 1).contiguous() # [bs, h, w, c]
    output = output + i_in

    k_rec = fft2c_new(output)
    k_rec = self.DataConsistency(k_rec, k_sampled, mask)
    output = ifft2c_new(k_rec)

    return output

class PositionalEncoding(nn.Module):

  def __init__(self, pe_dim=128, magnify=100.0):
    super(PositionalEncoding, self).__init__()
    # Compute the division term
    # note that the positional dim for x and y is equal to dim_of_pe/2
    self.dim = pe_dim
    self.div_term = nn.Parameter(torch.exp(torch.arange(0, self.dim/2, 2) * -(2 * math.log(10000.0) / self.dim)), requires_grad=False)      # [32]
    self.magnify = magnify

  def forward(self, p_norm):
    """
    given position:[bs, h*w*0.2, 2]

    return pe
    """

    p = p_norm * self.magnify  # normalized 到 [0, magnify] 之间

    no_batch = False
    if p.dim() == 2:    # no batch size
      no_batch = True
      p = p.unsqueeze(0)

    p_x = p[:, :, 0].unsqueeze(2)                            # [bs, h*w*0.2, 1]
    p_y = p[:, :, 1].unsqueeze(2)
    # assert p_x.shape[1] == p_y.shape[1]
    pe_x = torch.zeros(p_x.shape[0], p_x.shape[1], self.dim // 2).to(torch.device('cuda'))     # [bs, h*w*0.2, 64]
    pe_x[:, :, 0::2] = torch.sin(p_x * self.div_term)                       # [bs, h*w*0.2, 32]
    pe_x[:, :, 1::2] = torch.cos(p_x * self.div_term)


    pe_y = torch.zeros(p_x.shape[0], p_x.shape[1], self.dim // 2).to(torch.device('cuda'))     # [bs, h*w*0.2, 64]
    pe_y[:, :, 0::2] = torch.sin(p_y * self.div_term)
    pe_y[:, :, 1::2] = torch.cos(p_y * self.div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)                      # [bs, h*w*0.2, 128]

    if no_batch:
      pe = pe.squeeze(0)

    # [len, dim]
    return pe

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

