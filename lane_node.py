
import pickle
import time
import yaml
import cv2 # OpenCV library
import time
import numpy as np
import sys
from torchvision import transforms
import torch

sys.path.append("..")
from lane_detect_python import utils
from lane_detect_python import infer

PRINT_TIME_STATIC=True
FOR_SEG_VISULIZATION=False

CURVE_CHANGE_THRESHOLD = 0.1

class lane_detect_node():
    def __init__(self,torch_model_path='./runs/train/banqiao_cam8M3/weights/epoch_219.pt',model_input_size=(960,960)) -> None:
        super().__init__()

        # torch_model_path = './runs/train/banqiao_mix2m8m5/weights/epoch_119.pt'
        # torch_model_path = torch_model_path
        self.model = infer.load_model(torch_model_path)
        self.model_input_size = model_input_size

        #8M相机　在1280x736图片上选点.
        self.new_size=utils.NEW_SIZE
        x,y,X,Y = utils.x,utils.y,utils.X,utils.Y

        # # #8M相机　在960x540图片上的选点
        # self.new_size=(960,540)
        # x = [375, 296, 627, 561]
        # y = [355, 406, 407, 355]
        # X = [200, 200, 250, 250] #
        # Y = [200, 400, 400, 200]
        # X_offset,Y_offset = 250,200 #手动调整找到合适的点
        # X = [e + X_offset for e in X] #
        # Y = [e + Y_offset for e in Y]

        src = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]))
        dst = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]],[X[2], Y[2]], [X[3], Y[3]]]))
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

        self.current_frame=None
        self.ratio, self.pad_w,self.pad_h = None,None,None

        #多帧曲率情况
        self.pre_stable_curv = None
        self.curr_curv = None
        self.curv_change_thre = CURVE_CHANGE_THRESHOLD

    def smooth_curve(self):
        """
        如果曲率发生突变,则使用之前的稳定曲率.
        """
        if self.curr_curv/self.pre_stable_curv > self.curv_change_thre:
            return self.pre_stable_curv
        else:
            return self.curr_curv 

    def detect(self,img):
        self.current_frame = img
        self.current_frame = cv2.resize(img,self.new_size,interpolation=cv2.INTER_LINEAR)
        print('self.current_frame:{}'.format(self.current_frame.shape))

        tbegin = time.time()
        t0 = time.time()
        img,result_img = self.preprocess(self.current_frame )
        print('result_img:{}'.format(result_img.shape))
        t = (time.time() - t0)*1000
        if PRINT_TIME_STATIC:
            print('preprocess time: {} ms!'.format(t))
        
        t0 = time.time()
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img,0)
        img = img.half().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        result = self.model(img)
        t = (time.time() - t0)*1000
        if PRINT_TIME_STATIC:
            print('inference time: {} ms!'.format(t))

        t0 = time.time()
        fit_success,left_distance,right_distance,curvatures,fit_on_front_img = self.postprocess(result,result_img)
        t = (time.time() - t0)*1000
        if PRINT_TIME_STATIC:
            print('postprocess time: {} ms!'.format(t))

        t = (time.time() - tbegin)*1000
        if PRINT_TIME_STATIC:
            print('total process time: {} ms!'.format(t))

        if fit_success:
            print('fit line success')

        return fit_success,left_distance,right_distance,curvatures,fit_on_front_img

    def preprocess(self,img):
        img, self.ratio, (self.pad_w,self.pad_h) = utils.letterbox(img,self.model_input_size,auto=False, scaleup=False) 
        result_img = np.zeros_like(img) #letterbox
    
        img = img[:, :, ::-1].transpose(2, 0, 1)  # rgb chw
        img = np.ascontiguousarray(img)
        img = img/255.
        # img = img.astype('float32') #不影响推理结果

        return img,result_img

    def postprocess(self,result,result_img):
        for out in result:
            # print('out shape:{}'.format(out.shape))
            pass

        t0 = time.time()
        _,lane_pre = result[0],result[1]
        lane_pre_batch0 = lane_pre[0]
        lane_pre_batch0 = lane_pre_batch0.float().cpu()

        current_lane_pre = torch.sigmoid(lane_pre_batch0)
        current_lane_pre = current_lane_pre.reshape(self.model_input_size)
        current_lane_pre_mask = np.where(current_lane_pre>0.7)
        
        result_img[current_lane_pre_mask[0],current_lane_pre_mask[1],0] = 0
        result_img[current_lane_pre_mask[0],current_lane_pre_mask[1],1] = 0
        result_img[current_lane_pre_mask[0],current_lane_pre_mask[1],2] = 255 

        t = (time.time() - t0)*1000
        if PRINT_TIME_STATIC:
            print('model output to prob time: {} ms!'.format(t))


        #将模型的输出结果对应到原始图片上.
        """
        如果原始图片过大,这里可能要考虑下采样.同时初始化时的那个M也要同步修改.
        """
        t0 = time.time()
        result_on_origin_img = self.current_frame
        result_binary_img = np.zeros( (self.current_frame.shape[0],self.current_frame.shape[1]) )
        new_h = np.arange(self.current_frame.shape[0]) * self.ratio[0] + self.pad_h
        new_w = np.arange(self.current_frame.shape[1]) * self.ratio[0] + self.pad_w
        result_on_origin_img = result_img[new_h.astype(int)[:, np.newaxis], new_w.astype(int)]
        result_binary_img = result_img[new_h.astype(int)[:, np.newaxis], new_w.astype(int), 2]
        t = (time.time() - t0)*1000
        if PRINT_TIME_STATIC:
            print('convert model output to size before preprocess time: {} ms!'.format(t))

        t0 = time.time()
        fit_success,left_distance,right_distance,curvatures,fit_on_front_img = utils.post_process_on_bev(result_binary_img,self.current_frame,self.M,self.M_inv)
        t = (time.time() - t0)*1000
        if PRINT_TIME_STATIC:
            print('utils.post_process_on_bev time: {} ms!'.format(t))

        if FOR_SEG_VISULIZATION:
            print('result_binary_img shape:{}'.format(result_binary_img.shape))
            seg_on_front_img = self.current_frame
            lane_mask = (result_binary_img==255)
            seg_on_front_img[lane_mask] = (255, 0, 0)
            print('type(lane_mask):{},shape:{}'.format(type(lane_mask),lane_mask.shape))
            fit_on_front_img = seg_on_front_img

        return fit_success,left_distance,right_distance,curvatures,fit_on_front_img
    

def test_on_img(img_path):
    lane_node = lane_detect_node('./runs/train/banqiao_cam8m5/weights/epoch_299.pt',model_input_size=(544,960))
    img = cv2.imread(img_path)
    fit_success,left_distance,right_distance,curvatures,fit_on_front_img = lane_node.detect(img)
    cv2.imwrite('./fit_on_front_img.png',fit_on_front_img)

def export2onnx(pt_path,onnx_save_path):
    utils.export_onnx(pt_path,onnx_save_path)

def main(args=None):
    # lane_node = lane_detect_node('./runs/train/banqiao_cam8m4/weights/epoch_299.pt')
    lane_node = lane_detect_node('./runs/train/banqiao_cam8m5/weights/epoch_299.pt',model_input_size=(544,960))
    # img = cv2.imread('/home/autocore/work_sc/datasets/banqiao/20230418/road02/1681827555_642832180.png')
    # # img = cv2.imread('/home/autocore/work_sc/datasets/banqiao/20230418/road03/1681827679_42810484.png')
    # fit_success,left_distance,right_distance,curvatures,fit_on_front_img = lane_node.detect(img)
    # cv2.imwrite('./fit_on_front_img.png',fit_on_front_img)

    filelist=[]
    datadir = '/home/autocore/work_sc/datasets/banqiao/20230418/road03/'
    videoname = 'banqiao_cam8m_road03.avi'
    utils.get_file(datadir,filelist,sortKey=utils.sort_by_time)
    # print(filelist)

    fit_on_front_img_list = []
    video_size = None
    for i,img_path in enumerate(filelist):
        img = cv2.imread(img_path)
        fit_success,left_distance,right_distance,curvatures,fit_on_front_img = lane_node.detect(img)
        fit_on_front_img_list.append(fit_on_front_img)
        
        
        height, width, _ = fit_on_front_img.shape
        video_size = (width,height)
    
    print('saving {} to video:{}'.format(len(fit_on_front_img_list),videoname))
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), 15, video_size)
    for i in range(len(fit_on_front_img_list)):
        out.write(fit_on_front_img_list[i])
        
    out.release()



if __name__ == '__main__':
    test_on_img('/home/autocore/work_sc/datasets/banqiao/20230418/road02/1681827555_642832180.png')
    # export2onnx('./runs/train/banqiao_cam8m5/weights/epoch_299.pt','banqiao_8m_544x960.onnx')
    # main()