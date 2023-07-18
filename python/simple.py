import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine

import numpy as np
import cv2

import torch
from torch import nn


input_shape = (1, 3, 416, 720)


def do_trace(model, inp):
    model_trace = torch.jit.trace(model, inp)
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


script_module = torch.jit.load("/home/tekeliulas/workspace/tracking_athletes/models/dev_models/nano5_best.torchscript")

img = cv2.imread("/home/tekeliulas/workspace/tracking_athletes/data/example.jpg")

img = img.astype("float32")
img = cv2.resize(img, (720, 416))

img = np.transpose(img / 255.0, [2, 0, 1])
img = np.expand_dims(img, axis=0)


input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, shape_list)

# from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
# mod = ToMixedPrecision("float16")(mod)

# print(relay.transform.InferType()(mod))

target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

ctx = tvm.device(target, 0)
vm = VirtualMachine(vm_exec, ctx)


code, lib = executable.save()
# save and load the code and lib file.
tmp = tvm.contrib.utils.tempdir()
path_lib = tmp.relpath("lib.so")
lib.export_library(path_lib)
with open(tmp.relpath("code.ro"), "wb") as fo:
    fo.write(code)


vm.set_input("main", **{input_name: img})
tvm_res = vm.run()

with torch.no_grad():
    torch_res = model(torch.from_numpy(img))