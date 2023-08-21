import torch
import cv2
import numpy as np


#bgr order
COLOR_MAP = {
    'background' : (0,0,0),
    'white_solid' : (0,0,128),
    'white_dotted' : (0,128,0),
    'forward_arrow' : (128,0,128),
    'diversion_line' : (0,128,64),
    'other_arrow' : (0,0,64),
    'right_arrow' : (128,128,128)
}

# cls_names = ['background','white_solid','white_dotted','forward_arrow'] #需要加载到label的类别. 长度需要和cfg/multihead_multicls.yaml中的lane_cls_num保持一致.

cls_names = ['white_solid','white_dotted','forward_arrow'] #需要加载到label的类别. 长度需要和cfg/multihead_multicls.yaml中的lane_cls_num保持一致.


def imgpath2labelpath(img_path):
    label_path = img_path.replace('Image','Label/SegmentationClassPNG')
    return label_path

def load_label(label_path):
    # print(label_path)

    line_img = cv2.imread(label_path)

    nm_cls = len(cls_names)
    mask = torch.zeros((nm_cls, line_img.shape[0],line_img.shape[1])) #构建一个多分类 chw
    for i,cls in enumerate(cls_names): 
        h_idx,w_idx = None,None
        if cls == 'forward_arrow':
            # h_idx,w_idx = np.where(((line_img == COLOR_MAP['forward_arrow']) \
            #                         | (line_img == COLOR_MAP['other_arrow'])).all(axis=2))
            h_idx,w_idx = np.where((line_img == COLOR_MAP['forward_arrow']).all(axis=2)) 
            # print('all arrow--------------------')
        else:
            color = COLOR_MAP[cls]
            h_idx,w_idx = np.where((line_img == color).all(axis=2)) 
        
        mask[i,h_idx,w_idx] = 1 

    return mask

def load_label_letterbox(label_path,shape,scaleup):
    # if '1684228583_251883973.png' in label_path:
    #     print(label_path)

    line_img = cv2.imread(label_path)
    # print('line_img shape:{},shape:{}'.format(line_img.shape,shape))
    line_img, ratio, pad = letterbox(line_img, shape, auto=False, scaleup=scaleup)

    nm_cls = len(cls_names)
    mask = torch.zeros((nm_cls, line_img.shape[0],line_img.shape[1])) #构建一个多分类 chw
    for i,cls in enumerate(cls_names): 
        h_idx,w_idx = None,None
        if cls == 'forward_arrow':
            h_idx,w_idx = np.where((line_img == COLOR_MAP['forward_arrow']).all(axis=2)) 
            mask[i,h_idx,w_idx] = 1 

            h_idx,w_idx = np.where((line_img == COLOR_MAP['other_arrow']).all(axis=2)) 
            mask[i,h_idx,w_idx] = 1 

            h_idx,w_idx = np.where((line_img == COLOR_MAP['right_arrow']).all(axis=2)) 
            mask[i,h_idx,w_idx] = 1 
        else:
            color = COLOR_MAP[cls]
            h_idx,w_idx = np.where((line_img == color).all(axis=2)) 
            mask[i,h_idx,w_idx] = 1 
        
    return mask

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # print('dw:{},dh:{}'.format(dw,dh))
    return img, ratio, (dw, dh)