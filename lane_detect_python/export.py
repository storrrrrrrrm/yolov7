

import onnx
import torch
def export_onnx(pt_path,onnx_save_path,input_size):
    # change onnx input/output types
    # import onnx
    # onnx_model = onnx.load(ONNX_FILE_PATH)
    # graph = onnx_model.graph
    # in_type = getattr(graph.input[0], "type", None)
    # getattr(in_type, "tensor_type", None).elem_type = 10  # fp16
    # out_type = getattr(graph.output[0], "type", None)
    # getattr(out_type, "tensor_type", None).elem_type = 10  # fp16
    # onnx.save(onnx_model, ONNX_FILE_PATH)

    weigths = torch.load(pt_path) #model input:1280
   
    output_names = ['out0','out1','out2','out3','seg_out']
    
    # 加载模型 不清楚原因,只能在cpu上处理 torch版本:1.13.1+cu117
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = weigths['model'].float() #模型转换为fp32
    model = model.to(device)
    _ = model.eval()
 
    # Input
    h,w = input_size
    img = torch.zeros(1, 3, h,w) # image size(1,3,320,192) iDetection
    # img = img.half().to(device)
    img = img.to(device)
    print('img shape:{},is_cuda:{}'.format(img.shape,img.is_cuda))

    print('prepare to export*******')
    torch.onnx.export(model, img, onnx_save_path, verbose=False, opset_version=11, input_names=['images'],
                          output_names=output_names,dynamic_axes={
                            'images':{0:'batch_size'},
                            'out0':{0:'batch_size'},
                            'out1':{0:'batch_size'},
                            'out2':{0:'batch_size'},
                            'out3':{0:'batch_size'},
                            'seg_out':{0:'batch_size'}
                          })
    print('export done*******')

    # Checks
    onnx_model = onnx.load(onnx_save_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model


if __name__ == '__main__':
    # test_on_img('/home/autocore/work_sc/datasets/banqiao/20230418/road02/Image/1681827555_642832180.png')
    # export2onnx('./runs/train/banqiao_cam8m5/weights/epoch_299.pt','banqiao_8m_544x960.onnx')
    # export_onnx('./runs/train/banqiao_cam8M_1280x7362/weights/epoch_239.pt','banqiao_8m_736x1280.onnx',input_size=(736,1280))
    # export_onnx('/home/autocore/work_sc/yolov7/runs/train/banqiao_cam8M_1280x7362/weights/epoch_239.pt','banqiao_8m_736x1280_newname.onnx',input_size=(736,1280))

    export_onnx('/mnt/data/sc/yolov7/runs/train/multihead_multicls47/weights/epoch_000.pt','banqiao_multihead_multicls.onnx',input_size=(736,1280))
