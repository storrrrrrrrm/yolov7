import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# weigths = torch.load('./runs/train/yolov7209/weights/epoch_149.pt')
# weigths = torch.load('./runs/train/yolov7218/weights/epoch_079.pt') #model input:640
weigths = torch.load('./runs/train/yolov7221/weights/epoch_059.pt') #model input:1280
model = weigths['model']
# model = model.half().to(device)
model = model.to(device)
_ = model.eval()

# test_img='/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/Camera 6/170927_063811892_Camera_6.jpg'
def predict(test_img):
    img = cv2.imread(test_img) #hwc bgr
    result_on_origin_img = img.copy()#在做了letterbox后的图像上绘制
    img, ratio, (pad_w,pad_h) = letterbox(img,1280,auto=False, scaleup=False) 
    print('ratio:{},pad_w:{},pad_h:{}'.format(ratio,pad_w,pad_h))
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

        #在letterbox img上绘制
        result_img[current_lane_pre_mask[1],current_lane_pre_mask[2],0] = 0
        result_img[current_lane_pre_mask[1],current_lane_pre_mask[2],1] = 0
        result_img[current_lane_pre_mask[1],current_lane_pre_mask[2],2] = 255 

        #在原始图上绘制
        h_origin,w_origin,c = result_on_origin_img.shape
        for h in range(h_origin):
            for w in range(w_origin):
                new_h = int(h * ratio[0] + pad_h)
                new_w = int(w * ratio[0] + pad_w)
                result_on_origin_img[h,w,:] = result_img[new_h,new_w,:]

        return result_on_origin_img

def get_file(dirname,filelist=[],sortKey=None):
    for dirpath, dirname, filenames in os.walk(dirname):
        # if sort:
        #     filenames = sorted(filenames) #要排序　否则会乱序
        for filename in sorted(filenames,key=sortKey): 
            if filename.endswith('jpg') or filename.endswith('png'):
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
    # filelist=[]
    # rootdir = '/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/'
    # videoname = 'apollo_road02.avi'
    # get_file(rootdir,filelist)
    # save_video(filelist,videoname)

    # filelist=[]
    # rootdir = '/home/autocore/work_sc/datasets/andemen_street/20221205/out2'
    # videoname = 'andemen_road.avi'
    # def sortKey(filename):
    #     """
    #     /home/autocore/work_sc/datasets/andemen_street/20221205/out2/1046.jpg
    #     return 1046
    #     """
    #     idx = filename.split('/')[-1][:-4]
    #     idx = int(idx)      
    #     return idx 
    # get_file(rootdir,filelist,sortKey)
    # print(filelist)
    # save_video(filelist,videoname)

    # filelist=[]
    # rootdir = '/home/autocore/work_sc/datasets/banqiao'
    # videoname = 'banqiao.avi'
    # get_file(rootdir,filelist)
    # save_video(filelist,videoname)



main()
# test_img = '/home/autocore/work_sc/datasets/andemen_street/20221205/out2/111.jpg'
test_img = '/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/Camera 5/170927_063845516_Camera_5.jpg'
test_img = '/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/Camera 5/170927_063814371_Camera_5.jpg'

result_img = predict(test_img)
print('result_img:{}'.format(result_img.shape))
cv2.imwrite('./170927_063814371_Camera_5_prediction.png',result_img)










