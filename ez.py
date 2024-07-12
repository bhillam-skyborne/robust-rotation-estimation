import os
import sys
sys.path.append('./kaggle/input/raft-pytorch')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from glob import glob
from PIL import Image
from tqdm import tqdm

from raft.core.raft import RAFT
from raft.core.utils import flow_viz
from raft.core.utils.utils import InputPadder
from raft.config import RAFTConfig

from robust_estimation import RobustRotationEstimator

config = RAFTConfig(
	dropout=0,
	alternate_corr=False,
	small=False,
	mixed_precision=False
)

model = RAFT(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

weights_path = './kaggle/input/raft-pytorch/raft-sintel.pth'

ckpt = torch.load(weights_path, map_location=device)
model.to(device)
model.load_state_dict(ckpt)

#image_files = glob('./kaggle/input/raft-pytorch/raft/demo-frames/*.png')
#image_files = sorted(image_files)
#
#print(f'Found {len(image_files)} images')
#print(sorted(image_files))

def load_image(imfile, device):
	img = np.array(Image.open(imfile)).astype(np.uint8)
	img = torch.from_numpy(img).permute(2,0,1).float()
	return img[None].to(device)

def viz(img1, img2, flo):
	img1 = img1[0].permute(1,2,0).cpu().numpy()
	img2 = img2[0].permute(1,2,0).cpu().numpy()
	flo = flo[0].permute(1,2,0).cpu().numpy()

	flo = flow_viz.flow_to_image(flo)

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
	ax1.set_title('input image1')
	ax1.imshow(img1.astype(int))
	ax2.set_title('input image2')
	ax2.imshow(img2.astype(int))
	ax3.set_title('estimated optical flow')
	ax3.imshow(flo)
	plt.show()

model.eval()

#n_vis = 3

#for file1, file2 in tqdm(zip(image_files[:n_vis], image_files[1:1+n_vis])):
#	image1 = load_image(file1, device)
#	image2 = load_image(file2, device)
#
#	padder = InputPadder(image1.shape)
#	image1, image2 = padder.pad(image1, image2)
#
#	with torch.no_grad():
#		flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
#
#	viz(image1, image2, flow_up)

#video_file = './kaggle/input/nfl-impact-detection/train/57583_000002_Endzone.mp4'
video_file = '/mnt/c/Users/b.hillam/Downloads/GCS_Footage.mp4'

cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
timestamps = []

frames = []
while True:
	has_frame, image = cap.read()

	if has_frame:
		timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
		image = image[:, :, ::-1] #BGR -> RGB
		image = cv2.resize(image, (480, 270), interpolation = cv2.INTER_LINEAR)
#		print(f'im size: {image.shape}')
		frames.append(image)
	else:
		break
cap.release()
frames = np.stack(frames, axis=0)

print(f'frame shape: {frames.shape}')
#plt.imshow(frames[0])

rot_est = np.empty(shape=[0, 3])

n_vis = len(frames)

#erase file contents before append
with open('rot_est.csv', 'w') as file:
	pass

csv = open('rot_est.csv', 'a')

for i in range(n_vis):
	image1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().to(device)
	image2 = torch.from_numpy(frames[i+1]).permute(2, 0, 1).float().to(device)

#	print(f'im1 size: {image1.shape}')
#	print(f'im2 size: {image2.shape}')

	image1 = image1[None].to(device)
	image2 = image2[None].to(device)
	
	padder = InputPadder(image1.shape)
	image1, image2 = padder.pad(image1, image2)

#	print(f'im1 size: {image1.shape}')
#	print(f'im2 size: {image2.shape}')

	with torch.no_grad():
		flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

#	viz(image1, image2, flow_up)
	flow = np.transpose(flow_up.detach().cpu().numpy()[0,0:480,0:270,:], (1,2,0))
#	print(f'flow shape: {flow.shape}')

	h, w, _ = flow.shape

	args_f = 1655 / 4
	args_bin_size = 0.001
	args_max_angle = 0.15 #radians per frame, 0.4 = ~680 deg/sec @ 30 FPS
	args_spatial_step = 15
	
	rotation_estimator = RobustRotationEstimator(h, w, args_f, args_bin_size, args_max_angle, args_spatial_step)
	est = rotation_estimator.estimate(flow)

	print(f'{i}, {timestamps[i]}, {est[0]}, {est[1]}, {est[2]},')

	print(f'est shape {np.reshape(est, [1,3]).shape}')
	print(f'rotest shape {rot_est.shape}')
	rot_est = np.vstack([rot_est, est])
	np.savetxt(csv, np.reshape(est, [1,3]))
	

