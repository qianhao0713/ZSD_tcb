import onnx
from onnx.tools import update_model_dims
import numpy as np
import onnx.helper as helper
from onnx import shape_inference, TensorProto
onnx_model = onnx.load("../onnx/COCO_65_15.onnx")
graph = onnx_model.graph
all_input_tensors = graph.input
input_tensors = [(i, t) for i, t in enumerate(all_input_tensors) if t.name in ('img', 'img_shape', 'scale_factor')]
print(input_tensors)
# input_tensor_new = onnx.helper.make_tensor_value_info(name = input_tensor.name, elem_type = 1, 
#                                                       shape = [1, input_shape[1].dim_value, input_shape[2].dim_value, input_shape[3].dim_value])
