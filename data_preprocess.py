import numpy as np
from fftc import ifft2c_new, fft2c_new
import torch
import torch.nn as nn
import os
import h5py
import matplotlib.pylab as plt


def complex_abs(data):
    """
    Compute the squared absolute value of a complex tensor.
    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return ((data ** 2).sum(dim=-1) + 0.0).sqrt()

def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).
    RSS is computed assuming that dim is the coil dimension.
    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform
    Returns:
        The RSS value.
    """
    return torch.sqrt((data**2).sum(dim))


def save_image(np_data, path, type='np'):
  if type == 'torch':
    data = np_data.numpy()
  else:
    data = np_data
  if data.ndim == 3:
    plt.imshow(np.abs(data[:, :, 0] + 1j * data[:, :, 1]), cmap="gray")  # [h, w, c]
  elif data.ndim == 2:
    plt.imshow(np.abs(data), cmap="gray")
  plt.axis("off")
  plt.savefig(path)
  plt.close()

def save_k(np_data, path, type='np'):
  if type == 'torch':
    data = np_data.numpy()
  else:
    data = np_data
  plt.figure(dpi=300)
  plt.imshow(np.log(1 + np.abs(data[:, :, 0] + 1j * data[:, :, 1])), cmap="gray")
  plt.axis("off")
  plt.savefig(path)
  plt.close()

def normal_in_i(image):
  """
  normalized to N(0, 1)

  Args:
    data: image, array [h, w, 2]
  """
  mean = np.mean(image)
  std = np.std(image)
  tmp = (image - mean)/std
  return tmp

def concat_all_h5py(crop_size, root_path, save_path,mode,lr=False):
  """
  From h5 to npy
  
  Args:
      crop_size (int): cropped size of each slice, e.g. 320
      root_path (str): root path to h5 files
      save_path (str): save path to npy file
  """
  g = os.walk(root_path)
  data_list = []    # [num_volumes, 26, 640, 372, 2]
  count = 0
  for _, _, file_list in g:
    
    for file_name in file_list:
      # open file
      if file_name[-2:] != 'h5':
        continue
      id = file_name[-10:-3]
      file_path = os.path.join(root_path, file_name)
     
    
      try:
            
            # now volume is of shape (nslices,ncoils,h,w)
            
            
        volume = h5py.File(file_path)['kspace'][()]   # [26, 640, 372]
        
      except:
        print("Failed")
        continue
      # crop the size if needed
      # combining multicoil to single coil now:
        #create a for loop to iterate through the h5 file and access the individual slices, call the complex_abs fn as follows:
        
        
        # torch.sqrt(complex_abs_sq(data).sum(dim))
        #check output dim of the slice and figure out what crop size needs to be such that we get a 256 * 256 image
      width = volume.shape[2+1]  # adding +1 because we have another dimension 
      height = volume.shape[1+1]
      if width < crop_size or height < crop_size:
        print("height:",height,"width",width)
        print("no crop")
        
      else:
        count += 1
        print(mode,count)
        w_from = (width-crop_size)//2
        w_to = w_from + crop_size
        h_from = (height - crop_size) // 2
        h_to = h_from + crop_size
      # complex to double channel
      volume = np.stack([volume.real, volume.imag], axis=-1)    # [26, 640, 372, 2]
      n_s,n_c,h,w,_=volume.shape
    
      imgs_vols=torch.zeros((n_s,n_c,h,w))
      for s in range(n_s):             
          imgs_vols[s,:,:,:]=complex_abs(ifft2c_new(torch.from_numpy(volume[s,:,:,:,:])))
      imgs_rss=rss(imgs_vols,1)
      
      imgs_rss=torch.stack((imgs_rss,torch.zeros_like(imgs_rss))).permute(1,2,3,0)
      volume=fft2c_new(imgs_rss).numpy()
      resolution='lr' if lr==True else 'hr'
      fol_name='/images320_'+mode+ resolution +'/'
        
      if not os.path.isdir(root_path+fol_name):
            os.mkdir(root_path+fol_name)
            
      imag_volume = imgs_rss.numpy()

      #save_k(volume[-1], root_path + '/images320/' + id + 'BeforeCropNorm(K).png')  # check
      
      plt.imsave(root_path + fol_name + id + 'BeforeCropNorm(K).png',np.log(np.linalg.norm(volume[0,:,:,:],axis=-1)+1e-9),cmap='gray')
      # convert to image
     
      #save_image(imag_volume[-1], root_path+'/images320/'+id+'BeforeCropNorm.png')  # check
      plt.imsave(root_path +fol_name + id + 'BeforeCropNorm.png',np.abs(imag_volume[0,:,:,0]), cmap='gray')
      # crop
      #All ground truth images are cropped to the central 320 ×320 pixel region to compensate for readout-direction oversampling that is standard in clinical MR examinations.
      imag_volume = imag_volume[:, h_from:h_to, w_from:w_to, :]
      #print("BEFORE",imag_volume.shape)
      if lr==True:
        imag_volume=down_sample_I(imag_volume) 
      #print("AFTER",imag_volume.shape)
      # normalize each sample
      for i in range(imag_volume.shape[0]):
        imag_volume[i, :, :, :] = normal_in_i(imag_volume[i, :, :, :])
      #save_image(imag_volume[-1], root_path + '/images320/' + id + 'AfterCropNorm.png')  # check
    
      plt.imsave(root_path + fol_name+ id + 'AfterCropNorm.png',np.abs(imag_volume[0,:,:,0]), cmap='gray')
      # back to k space
      k_volume = fft2c_new(torch.from_numpy(imag_volume)).numpy()
      #save_k(k_volume[-1], root_path + '/images320/' + id + 'AfterCropNorm(K).png')  # check
      plt.imsave(root_path + fol_name + id + 'AfterCropNorm(K).png',np.log(np.linalg.norm(volume[0,:,:,:],axis=-1)+1e-9),cmap='gray')
      data_list.append(k_volume)
      print(len(data_list),"len")

  data_concat = np.concatenate(data_list, axis=0)   # [num_volumes*volume_len, h, w, 2]
  print(data_concat.shape,"DONE PROCESSING "+mode+" IMAGES") 
  np.save(save_path, data_concat)

def down_sample_I(np_I_data, scale=2):
  """
  Downsample image
  
  Args:
    np_I_data: image [N, h, w, c]
    scale: down sample ratio

  Returns: down sampled image [N, h/2, w/2, c]
  """
  m = nn.AvgPool2d(scale, stride=scale)
  i = torch.from_numpy(np_I_data)  # [n, 256, 256, 2]
  #save_image(i[20], 'image_before_ds.png', 'torch') # check

  i = i.permute(0, 3, 1, 2)
  ds_i = m(i)

  ds_i = ds_i.permute(0, 2, 3, 1)
  #save_image(ds_i[20], 'image_after_ds.png', 'torch') # check

  return ds_i.numpy()

if __name__ == "__main__":
    modes=['train','test','valid']
    lr_list=[False,True]
    for lr in lr_list:
        for mode in modes:
            resolution='lr' if lr else 'hr'
            root_path = "/home/mainuser/datadrive/batch_multicoil_train_batch_0/multicoil_"+mode+'/'
            crop_size = 256
            save_path = "/home/mainuser/datadrive/predata/data/"+mode+"/"+resolution+'/'+"numpy_"+mode+"_"+resolution+str(crop_size)+".npy"
            
            concat_all_h5py(crop_size, root_path, save_path,mode,lr)
