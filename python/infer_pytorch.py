import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
import cv2
import torch.nn.functional as F

def run_inference(image, gen_module):
    image = cv2.resize(image, (128, 256))
    image = image.transpose((2, 0, 1))
    image = image.astype("float32")
    image /= 255.0
    print(image.shape)
    image = np.expand_dims(image, axis=0)

    # Set the input
    gen_module.set_input("input0", image)

    # Run the model
    gen_module.run()

    # Get the output
    out = gen_module.get_output(0).asnumpy()
    return out

# Load the model
loaded_lib = tvm.runtime.load_module("../models/dep_models/osnet1.so")
gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib["default"](tvm.cpu(0)))

# Load and process the image
image = cv2.imread("../output/cropped/188_5.png")
image2 = cv2.imread("../output/cropped/189_0.png")

out1 = run_inference(image, gen_module)
out2 = run_inference(image2, gen_module)
dist = np.mean(np.abs(out2-out1))
print(dist) # Should be (1, 128)
