import numpy as np
import torch
import argparse
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from utils.utils_test import rggb_to_rgb, p4ip_wrapper_pad, img_register
from utils.utils_torch import conv_fft, img_to_tens, scalar_to_tens
from utils.utils_deblur import pad, crop

from models.network_p4ip import P4IP_Net

parser = argparse.ArgumentParser(description='Test on Real Data')
parser.add_argument('--idx', type=int, default=10, help='index of real data file [0,29]')
args = parser.parse_args()

DIR = 'data/real_data/lux5'
MODEL_FILE = 'model_zoo/p4ip_100epoch.pth'
IDX = args.idx
IDX_CLEAN = int(IDX/3)

# Load data-files
y = np.load(DIR+'/cut'+str(IDX)+'.npy') 			# Noisy image
k = np.load(DIR+'/kernel'+str(IDX)+'.npy') 			# blur kernel
# Clipping negative values and normalization of kernel
y = np.clip(y.astype(np.float32),0,np.inf); k = np.clip(k.astype(np.float32),0,np.inf)
k /= np.sum(np.ravel(k))
k = pad(k , [65,65])
if IDX in [15,20,24]:
	MODE = 'RGGB'
else:
	MODE = 'BGGR'


"""
Load the network
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p4ip = P4IP_Net(); p4ip.load_state_dict(torch.load(MODEL_FILE)); 
p4ip.to(device); p4ip.eval()


"""
Prepare the variables for input to network
"""
y_list = []
y_list.append(y[0::2, 0::2])
y_list.append(y[0::2, 1::2])
y_list.append(y[1::2, 0::2])
y_list.append(y[1::2, 1::2])
H, W = np.shape(y)
M_list = []
for y in y_list:
	M_hat = np.mean(np.ravel(y))/0.33
	M_list.append(M_hat)

with torch.no_grad():	
	rggb = []
	for y, M in zip(y_list, M_list):
		x_rec = p4ip_wrapper_pad(y, k, M, p4ip)
		x_rec = x_rec*M
		rggb.append(x_rec)	
	x_p4ip = rggb_to_rgb(rggb, H, W, MODE)
	y = rggb_to_rgb(y_list, H, W, MODE)
den_arr = np.load('data/real_data/denominator_data.npy')
den = den_arr[IDX_CLEAN]
x_p4ip = x_p4ip.astype(np.float32)*255/den
y = y.astype(np.float32)*255/den
cv2.imwrite('results/p4ip_'+str(IDX)+'.png', x_p4ip )
cv2.imwrite('results/y_'+str(IDX)+'.png', y )

# Load images in CV2 format for image registration and comparison
x_p4ip_cv = cv2.imread('results/p4ip_'+str(IDX)+'.png')
x_gt_cv = cv2.imread(DIR+'_clean/'+str(IDX_CLEAN)+'.png')	# clean image
y_cv  = cv2.imread('results/y_'+str(IDX)+'.png')

im_true_register, im_estimated =  img_register(x_gt_cv, x_p4ip_cv)
plt.figure(figsize=(10,6))
plt.subplot(1,3,1); plt.imshow(np.flip(im_true_register, axis=2)); plt.axis('off')
plt.title('True Image, After Registration')
plt.subplot(1,3,2); plt.imshow(np.flip(y_cv, axis=2)); plt.axis('off')
plt.title('Noisy Blurred Image')
plt.subplot(1,3,3); plt.imshow(np.flip(im_estimated, axis=2)); plt.axis('off')
plt.title('Reconstructed Image')
plt.savefig('results/demo_real.png', bbox_inches='tight', pad_inches=0.05)
plt.show()

