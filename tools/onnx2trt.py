import argparse
from load_plugin_lib import load_plugin_lib
load_plugin_lib()
import tensorrt as trt

def convert(src, dst):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    registry = trt.get_plugin_registry()
    plugin_creator = registry.get_plugin_creator("roi_align", "1", "")
    builder = trt.Builder(logger)
    profile = builder.create_optimization_profile()
    calib_profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    workspace = 20
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(src):
        raise RuntimeError(f'failed to load ONNX file: {src}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    # dynamic_input = {
    #     "img": [(1,3,340,340), (1,3,800,1216), (1,3,1400,1400)]
    # }
    
    dynamic_input = {
        "img": [(1,3,340,340), (1,3,768,1344), (1,3,1400,1400)]
    }
    # dynamic_input_value = {
    #     "img_shape": [(340,340,3), (768, 1344,3), (1400, 1400, 3)]
    # }
    for inp in inputs:
        if inp.name in dynamic_input:
            profile.set_shape(inp.name, *dynamic_input[inp.name])
            calib_profile.set_shape(inp.name, dynamic_input[inp.name][1], dynamic_input[inp.name][1], dynamic_input[inp.name][1])
        # if inp.name in dynamic_input_value:
        #     profile.set_shape_input(inp.name, *dynamic_input_value[inp.name])
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    config.add_optimization_profile(profile)
    config.set_calibration_profile(calib_profile)
    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')
    half = True
    print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(dst, 'wb') as t:
        t.write(engine.serialize())


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src pytorch model path')
    parser.add_argument('dst', help='dst onnx model path')

    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
