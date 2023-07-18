"""
ReIdentification Feature Extractor Module

Main task is to extract features of N size from an image
"""
import torch
from torchvision.transforms import (
	Resize, Compose, ToTensor, Normalize, ToPILImage
)
from PIL import Image
import numpy as np
import torch.nn.functional as F
import cv2

class ReID:
	"""
	ReID class is used for extracting features out of images
	"""

	def __init__(self, model_path, input_size=(256, 128)):
		"""

		@param model_path: Path of reid jit model
		@param input_size: Reid models input size (h, w)
		"""
		self.height = input_size[0]
		self.width = input_size[1]
		self.norm_mean = [0.485, 0.456, 0.406]
		self.norm_std = [0.229, 0.224, 0.225]
		self.model = torch.load(model_path).eval()
		self.is_cuda = torch.cuda.is_available()
		if self.is_cuda:
			self.model = self.model.cuda()
		self.transform = Compose([
			ToPILImage(),
			Resize((self.height, self.width)),
			ToTensor(),
			Normalize(mean=self.norm_mean, std=self.norm_std),
		])

	def __call__(self, image):
		"""

		@param image: np.array of cropped image
		@return: torch.tensor of n sized vector
		"""
		input_img = self.transform(image)
		if self.is_cuda:
			input_img = input_img.cuda()
		input_img = input_img.unsqueeze(0)
		out_features = self.model(input_img)
		return out_features

modelpath = "../models/dev_models/osnet1_jit.pth"
model = torch.jit.load(modelpath)
model2 = ReID(modelpath)
img1 = cv2.imread("../output/cropped/188_5.png")
img2 = cv2.imread("../output/cropped/189_0.png")

out1 = model2(img1)
out2 = model2(img2)
dist = F.pairwise_distance(out2, out1, keepdim=True)

out1 = out1.cpu().detach().numpy()
out2 = out2.cpu().detach().numpy()
dist2 = np.mean(np.abs(out2-out1))
print(dist)
print(dist2)
print(out1[0,:5])

