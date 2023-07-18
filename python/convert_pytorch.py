import torch
import tvm
from tvm import te
from tvm import relay

# Load pre-trained resnet50 model from torchvision
# resnet50 = models.resnet50(pretrained=True)
# resnet50 = resnet50.eval()
scripted_model = torch.load("/home/tekeliulas/workspace/tracking_athletes/models/dev_models/best.torchscript")
# Create a sample input
input_shape = [1, 3, 416, 720]
input_data = torch.randn(input_shape)
# scripted_model = torch.jit.trace(model, input_data).eval()

# Convert the PyTorch model to TVM Relay format
input_name = 'input0'
shape_list = [(input_name, input_shape)]
print("relay started")
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
print("relay ended")
# Now you have TVM model (mod) and parameters (params)
# You can now compile the TVM model and run it on any supported backend

# For example, compile the model for CPU
print("build started")
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
print("build ended")

# Save the compiled module
lib.export_library('yolov8n.so')
