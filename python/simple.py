import tvm
from tvm import relay
from ctypes import POINTER, c_int, c_char_p, c_void_p
from tvm.relay.testing.darknet import __darknetffi__
import numpy as np


libso = "/home/rap018/workspace_ulas/misc/assignment/libdarknet.so"
# Load the Darknet model
darknetlib = __darknetffi__.dlopen(libso)

cfg = "/home/rap018/workspace_ulas/misc/assignment/models/dev_models/pd.cfg"
weights = "/home/rap018/workspace_ulas/misc/assignment/models/dev_models/pd.weights"

net = darknetlib.load_network(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
print("after")
dtype = 'float32'
batch_size = 1
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape = {'data': data.shape}
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

# Now compile the model
target = 'llvm'
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)

# Save the compiled module
lib.export_library('compiled.so')
