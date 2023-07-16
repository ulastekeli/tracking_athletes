# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile YOLO-V2 and YOLO-V3 in DarkNet Models
=============================================
**Author**: `Siju Samuel <https://siju-samuel.github.io/>`_

This article is an introductory tutorial to deploy darknet models with TVM.
All the required models and libraries will be downloaded from the internet by the script.
This script runs the YOLO-V2 and YOLO-V3 Model with the bounding boxes
Darknet parsing have dependancy with CFFI and CV2 library
Please install CFFI and CV2 before executing this script

.. code-block:: bash

  pip install cffi
  pip install opencv-python
"""

# numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys

# tvm, relay
import tvm
from tvm import te
from tvm import relay
from ctypes import *
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet
import cv2


def convert_image(image):
	"""Convert the image with numpy."""
	imagex = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imagex = np.array(imagex)
	imagex = imagex.transpose((2, 0, 1))
	imagex = np.divide(imagex, 255.0)
	imagex = np.flip(imagex, 0)
	return imagex


def load_image_color(test_image):
	"""To load the image using opencv api and do preprocessing."""
	imagex = cv2.imread(test_image)
	return convert_image(imagex)


def _letterbox_image(img, w_in, h_in):
	"""To get the image in boxed format."""
	imh, imw, imc = img.shape
	if (w_in / imw) < (h_in / imh):
		new_w = w_in
		new_h = imh * w_in // imw
	else:
		new_h = h_in
		new_w = imw * h_in // imh
	dim = (new_w, new_h)
	# Default interpolation method is INTER_LINEAR
	# Other methods are INTER_AREA, INTER_NEAREST, INTER_CUBIC and INTER_LANCZOS4
	# For more information see:
	# https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
	resized = cv2.resize(src=img, dsize=dim, interpolation=cv2.INTER_CUBIC)
	resized = convert_image(resized)
	boxed = np.full((imc, h_in, w_in), 0.5, dtype=float)
	_, resizedh, resizedw = resized.shape
	boxed[
	:,
	int((h_in - new_h) / 2) : int((h_in - new_h) / 2) + resizedh,
	int((w_in - new_w) / 2) : int((w_in - new_w) / 2) + resizedw,
	] = resized
	return boxed


def load_image(img, resize_width, resize_height):
	"""Load the image and convert to the darknet model format.
	The image processing of darknet is different from normal.
	Parameters
	----------
	image : string
		The image file name with path

	resize_width : integer
		The width to which the image needs to be resized

	resize_height : integer
		The height to which the image needs to be resized

	Returns
	-------
	img : Float array
		Array of processed image
	"""
	imagex = cv2.imread(img)
	return _letterbox_image(imagex, resize_width, resize_height)


######################################################################
# Choose the model
# -----------------------
# Models are: 'yolov2', 'yolov3' or 'yolov3-tiny'


######################################################################
# Download required files
# -----------------------
# Download cfg and weights file if first time.
REPO_URL = "https://github.com/dmlc/web-data/blob/main/darknet/"


# cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
# weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")


cfg_path = "/home/rap018/workspace_ulas/misc/assignment/models/dev_models/pd.cfg"
weights_path = "/home/rap018/workspace_ulas/misc/assignment/models/dev_models/pd.weights"


libso = "/home/rap018/workspace_ulas/misc/assignment/libdarknet.so"
DARKNET_LIB = __darknetffi__.dlopen(libso)
net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
dtype = "float32"
batch_size = 1

data = np.empty([batch_size, 3, 544, 960], dtype)
shape_dict = {"data": data.shape}
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape = {"data": data.shape}
print("Compiling the model...")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

[neth, netw] = shape["data"][2:]  # Current image shape is 608x608


