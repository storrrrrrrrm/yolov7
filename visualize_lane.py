import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('./runs/train/yolov7209/weights/epoch_149.pt')
model = weigths['model']
# model = model.half().to(device)
model = model.to(device)
_ = model.eval()

# test_img='/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/Camera 6/170927_063811892_Camera_6.jpg'
def predict(test_img):
    img = cv2.imread(test_img) #hwc bgr
    img, _, _ = letterbox(img,640,auto=False, scaleup=False) 
    result_img = img.copy()#在做了letterbox后的图像上绘制
    img = img[:, :, ::-1].transpose(2, 0, 1)  # rgb chw
    img = np.ascontiguousarray(img)
    img = img/255.

    img = torch.from_numpy(img)
    img = torch.unsqueeze(img,0)
    img = img.half().to(device)

    output = model(img)
    _,lane_pre = output[0],output[1] #lane 2x640x640
    lane_pre = lane_pre.float().cpu() #bchw

    b = lane_pre.shape[0]
    for i in range(b):  
        current_lane_pre = torch.sigmoid(lane_pre[i,...])
        current_lane_pre_mask = np.where(current_lane_pre>0.7)
        pre_lane_num = len(current_lane_pre_mask[1])
        print('当前检测出车道点个数:{}'.format(pre_lane_num))
        print(current_lane_pre_mask[1])
        print(current_lane_pre_mask[2])

        result_img[current_lane_pre_mask[1],current_lane_pre_mask[2],0] = 0
        result_img[current_lane_pre_mask[1],current_lane_pre_mask[2],1] = 0
        result_img[current_lane_pre_mask[1],current_lane_pre_mask[2],2] = 255 

        # cv2.imwrite('./prediction.png',result_img)

        return result_img

def get_file(dirname,filelist=[]):
    for dirpath, dirname, filenames in os.walk(dirname):
        for filename in sorted(filenames): #要排序　否则会乱序
            if filename.endswith('jpg'):
                fullpath = os.path.join(dirpath, filename)
                filelist.append(fullpath)

def save_video(filelist,videoname):
    result_img_list=[]
    size=None
    for i,f in enumerate(filelist):
        print('predict on {} image'.format(i))
        result_img = predict(f)
        height, width, layers = result_img.shape
        size = (width,height)
        result_img_list.append(result_img)
        # cv2.imwrite('./prediction_results/prediction{}.png'.format(i),result_img)
        # if i > 10:
        #     break
    try:
        os.remove(videoname)
    except FileNotFoundError:
        print("File is not present in the system.going to create {}".format(videoname))

    print('saving {} to video'.format(len(result_img_list)))
    print('size:{}'.format(size))
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(result_img_list)):
        # print('{}'.format(result_img_list[i].shape))
        out.write(result_img_list[i])
        # print('save {}'.format(filelist[i]))
        
    out.release()

def main():
    filelist=[]
    rootdir = '/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/'
    get_file(rootdir,filelist)

    videoname = 'apollo_road02.avi'
    save_video(filelist,videoname)

main()








