import cv2
import numpy as np

#bgr order
COLOR_MAP = {
    'white_solid' : (0,0,128),
    'white_dotted' : (0,128,0),
    'forward_arrow' : (128,0,128),
    'diversion_line' : (0,128,64),
    'other_arrow' : (0,0,64),
    'right_arrow' : (128,128,128)
}

class ImgInfo():
    def __init__(self,cls_names):
        super().__init__()

        self.cls_names = cls_names
        self.cls_exist = [False] * len(cls_names)
        self.pixel_num = [0] * len(cls_names)

    def print(self):
        for i,cls in enumerate(self.cls_names):
             print('cls:{},pixel_num:{},self.cls_exist:{}'.format(cls,self.pixel_num[i],self.cls_exist[i]))

def analysis(img_path):
    line_img = cv2.imread(img_path)

    cls_names = [k for k in COLOR_MAP.keys()]
    info = ImgInfo(cls_names)

    for i,cls in enumerate(cls_names): 
        color = COLOR_MAP[cls]
        h_idx,w_idx = np.where((line_img == color).all(axis=2)) 

        info.pixel_num[i] = len(h_idx)
        if len(h_idx) != 0:
            info.cls_exist[i] = True
        
    info.print()

    return info


def imgpath2labelpath(img_path):
    label_path = img_path.replace('Image','Label/SegmentationClassPNG')
    return label_path

class TrainSetInfo():
    def __init__(self,cls_names):
        super().__init__()

        self.cls_names = cls_names
        self.img_num = [0] * len(cls_names) #img num which has cls
        self.pixel_num = [0] * len(cls_names) #all pixel_nums in traindaset

    def print(self):
        for i,cls in enumerate(self.cls_names):
            print('cls:{},img_num:{},pixel_num:{}'.format(cls,self.img_num[i],self.pixel_num[i]))

def analysis_infos(infos):
    cls_names = [k for k in COLOR_MAP.keys()]
    trainset_info = TrainSetInfo(cls_names) 
    
    for i,cls in enumerate(cls_names): 
        for img_info in infos:
            if img_info.cls_exist[i]:
                trainset_info.img_num[i] = trainset_info.img_num[i] + 1 
                trainset_info.pixel_num[i] = trainset_info.pixel_num[i] + img_info.pixel_num[i]
    
    trainset_info.print()
    return trainset_info

def analysis_traindataset(traintxt_path):
    infos = []
    with open(traintxt_path) as f:
        for trainpath in f.readlines():
            if trainpath[-1] == '\n':
                trainpath = trainpath[:-1] # remove last '\n'
            label_path = imgpath2labelpath(trainpath)
            info = analysis(label_path)

            print(label_path)

            infos.append(info)

    analysis_infos(infos)
    return infos




if __name__ == '__main__':
    # img_path = '/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Label/SegmentationClassPNG/1684228583_251883973.png'
    # analysis(img_path)

    analysis_traindataset('/mnt/data/sc/yolov7/banqiao/banqiao_lane_seg.txt')