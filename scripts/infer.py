import tensorrt as trt
import numpy as np
import os
import cv2
import onnx
import torch

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(flags=EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file:{}'.format(onnx_path))
            for error in range(parser.num_errors):
                print (parser.get_error(error))
    
    serialized_engine = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        print('save engine to {}'.format(engine_path))

def onnx2trtengine(onnx_path,engine_path):
    model = onnx.load(onnx_path)
    engine = trt.utils.onnx2trt(
        logger=trt.Logger(trt.Logger.WARNING),
        model=model,
        max_batch_size=1,
        max_workspace_size=1 << 30,
        fp16_mode=True
    )

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load(model_path) 

    model = weigths['model']
    model = model.to(device)
    _ = model.eval()

    return model
        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # build_engine('banqiao.onnx','banqiao.engine')
    # build_engine('../config/banqiao_mix2m8m.onnx','../config/banqiao.engine')
    # test_infer()
    build_engine('../banqiao_8m_736x1280_multicls.onnx','../banqiao_8m_736x1280_multicls.engine')
