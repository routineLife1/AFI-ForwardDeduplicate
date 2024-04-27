# class WarpLayer warps image x based on optical flow flo.
import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ForwardWarp(nn.Module):
	"""docstring for WarpLayer"""
	def __init__(self,):
		super(ForwardWarp, self).__init__()
	

	def forward(self, img, flo):
		"""
			-img: image (N, C, H, W)
			-flo: optical flow (N, 2, H, W)
			elements of flo is in [0, H] and [0, W] for dx, dy

		"""
		

		# (x1, y1)		(x1, y2)
		# +---------------+
		# |				  |
		# |	o(x, y) 	  |
		# |				  |
		# |				  |
		# |				  |
		# |				  |
		# +---------------+
		# (x2, y1)		(x2, y2)


		N, C, _, _ = img.size()
		
		# translate start-point optical flow to end-point optical flow
		y = flo[:, 0:1 :, :]
		x = flo[:, 1:2, :, :]

		x = x.repeat(1, C, 1, 1)
		y = y.repeat(1, C, 1, 1)
	
		# Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
		# (1.5,1.5)
		# (1,1) (1,2) (2,1) (2,2)
		x1 = torch.floor(x)
		x2 = x1 + 1
		y1 = torch.floor(y)
		y2 = y1 + 1

		# firstly, get bilinear weights
		w11, w12, w21, w22 = self.get_bilinear_weights(x, y, x1, x2, y1, y2)

		# secondly, sample each weighted corner 
		img11 = self.sample_one(img, x1, y1, w11)
		img12 = self.sample_one(img, x1, y2, w12)
		img21 = self.sample_one(img, x2, y1, w21)
		img22 = self.sample_one(img, x2, y2, w22)


		imgw = img11 + img12 + img21 + img22

		return imgw

	def get_gaussian_kernel(self, kernel_size=3, sigma=2, channels=3):
		# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
		x_coord = torch.arange(kernel_size)
		x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
		y_grid = x_grid.t()
		xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

		mean = (kernel_size - 1)/2.
		variance = sigma**2.

		# Calculate the 2-dimensional gaussian kernel which is
		# the product of two gaussian distributions for two different
		# variables (in this case called x and y)
		gaussian_kernel = (1./(2.*math.pi*variance)) *\
						torch.exp(
							-torch.sum((xy_grid - mean)**2., dim=-1) /\
							(2*variance)
						)

		# Make sure sum of values in gaussian kernel equals 1.
		gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

		# Reshape to 2d depthwise convolutional weight
		gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
		gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

		gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

		gaussian_filter.weight.data = gaussian_kernel
		gaussian_filter.weight.requires_grad = False
		
		return gaussian_filter

	def get_bilinear_weights(self, x, y, x1, x2, y1, y2):
		w11 = (x2 - x) * (y2 - y)
		w12 = (x2 - x) * (y - y1)
		w21 = (x - x1) * (y2 - y)
		w22 = (x - x1) * (y - y1)
		
		return w11, w12, w21, w22


	def sample_one(self, img, shiftx, shifty, weight):
		"""
		Input:
			-img (N, C, H, W)
			-shiftx, shifty (N, c, H, W)
		"""

		N, C, H, W = img.size()
		use_gpu = img.is_cuda

		# flatten all (all restored as Tensors)
		flat_shiftx = shiftx.view(-1)
		flat_shifty = shifty.view(-1)
		if use_gpu:
			flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].cuda().long().repeat(N, C, 1, W).view(-1)
			flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].cuda().long().repeat(N, C, H, 1).view(-1)
		else:
			flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].long().repeat(N, C, 1, W).view(-1)
			flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].long().repeat(N, C, H, 1).view(-1)
		flat_weight = weight.view(-1)
		flat_img = img.view(-1)


		# The corresponding positions in I1
		if use_gpu:
			idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).long().cuda().repeat(1, C, H, W).view(-1)
			idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).long().cuda().repeat(N, 1, H, W).view(-1)
		else:
			idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).long().repeat(1, C, H, W).view(-1)
			idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).long().repeat(N, 1, H, W).view(-1)			
		# ttype = flat_basex.type()
		idxx = flat_shiftx.long() + flat_basex
		idxy = flat_shifty.long() + flat_basey


		# recording the inside part the shifted
		mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)
		# mask = mask

		# Mask off points out of boundaries
		
		# index = torch.arange(mask.size(0)).type(torch.cuda.LongTensor).cuda()[mask]
		ids = (idxn*C*H*W + idxc*H*W + idxx*W + idxy)

		if use_gpu:
			ids_mask = torch.masked_select(ids, mask).clone().cuda()
		else:
			ids_mask = torch.masked_select(ids, mask).clone()
		# del flat_basex, flat_basey, idxx, idxy, idxn, idxc, ids

		# ids = Variable((idxn*C*H*W + idxc*H*W + idxx*W + idxy))[mask].cuda()
		# ids = Variable(torch.gather(ids, 0, index)).cuda()
		# idxx = Variable(idxx[mask]).cuda()
		# idxy = Variable(idxy[mask]).cuda()
		# idxn = Variable(idxn[mask]).cuda()
		# idxc = Variable(idxc[mask]).cuda()

		#(zero part - gt) -> difference
		#difference back propagate -> No influence! Whether we do need mask? mask?
		# put (add) them together
		if use_gpu:
			img_warp = torch.zeros([N*C*H*W, ]).cuda().to(img.dtype)
		else:
			img_warp = torch.zeros([N*C*H*W, ]).to(img.dtype)
		img_warp.put_(ids_mask, torch.masked_select(flat_img*flat_weight, mask), accumulate=True)
		
		# if use_gpu:
		# 	one_warp = torch.zeros([N*C*H*W, ]).cuda()
		# else:
		# 	one_warp = torch.zeros([N*C*H*W, ])
		# one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

		# calculate zero mask


		return img_warp.view(N, C, H, W)

	