import torch
from copy import deepcopy
import struct

def save_bytes(bytes_list, filepath):
    '''Takes in a list of ints between 0 and 255, encodes them as bytes in a file at the specified path. Encoding should be compact, so that the file is not more bytes than the length of the list.'''
    with open(filepath, 'wb') as file:
        # Pack each integer into a single byte
        packed_bytes = struct.pack('B' * len(bytes_list), *bytes_list)
        file.write(packed_bytes)

def load_bytes(filepath):
    '''Takes in the path to a file, and returns the bytes in that file as a list of ints between 0 and 255.'''
    with open(filepath, 'rb') as file:
        # Read all bytes from the file
        file_bytes = file.read()
        # Unpack bytes into a tuple of integers
        return list(struct.unpack('B' * len(file_bytes), file_bytes))

def np_to_torch_img(img_np):
    img_pt = torch.tensor(img_np.astype('float') / 255)
    img_pt = img_pt.permute(2,0,1).unsqueeze(0).half().to('cuda')
    return img_pt

def torch_to_np_img(img):
    return (img[0].permute(1,2,0).clip(0,1).detach().cpu().numpy() * 255).astype('uint8')

def get_psnr(recon_pt, gt_pt):
    from skimage.metrics import peak_signal_noise_ratio
    return peak_signal_noise_ratio(torch_to_np_img(recon_pt), torch_to_np_img(gt_pt))

def find_first_leq_index(timesteps, recon_t):
    for i, t in enumerate(timesteps):
        if t <= recon_t:
            return i
    return -1