import torch
import cv2
import numpy as np

#bgr order
COLOR_MAP = {
    'white_solid' : (0,0,128),
    'white_dotted' : (0,128,0),
    'forward_arrow' : (128,0,128),
    'diversion_line' : (0,128,64)
}

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
        color = COLOR_MAP[cls]
        h_idx,w_idx = np.where((line_img == color).all(axis=2)) 
        mask[i,h_idx,w_idx] = 1 

    return mask