
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
from lane_detect_python import export

PRINT_TIME_STATIC=False
FOR_SEG_VISULIZATION=True #为true时将postprocess返回的fit_on_front_img替换为模型分割的结果

CURVE_CHANGE_THRESHOLD = 0.1
MAX_FRAMES_NUM = 10 
REFER_PAST_FRAMES=False #曲率变化过大的时候是否参考前几帧的结果

class LaneDetectionResults():
    def __init__(self):
        super().__init__()

        self.left_fit = None
        self.right_fit = None
        self.curvs = None
    
    def print_details(self):
        print('left_fit:{}'.format(self.left_fit))
        print('right_fit:{}'.format(self.right_fit))
        print('curvs:{}'.format(self.curvs))

class lane_detect_node():
    def __init__(self,torch_model_path='./runs/train/banqiao_cam8M3/weights/epoch_219.pt',model_input_size=(960,960)) -> None:
        super().__init__()

        # torch_model_path = './runs/train/banqiao_mix2m8m5/weights/epoch_119.pt'
        # torch_model_path = torch_model_path
        self.model = infer.load_model(torch_model_path)
        self.model_input_size = model_input_size

        self.new_size=utils.NEW_SIZE
   
        self.M = utils.H
        self.M_inv = utils.H_inv

        self.current_frame=None
        self.ratio, self.pad_w,self.pad_h = None,None,None

        #连续多真的检测结果       
        self.detection_results = []

        #当前选择使用的检测结果.不一定是当前帧的检测结果.比如当前帧曲率突变,则选择过去n帧的结果均值.
        self.prefer_detection_result = LaneDetectionResults()

    def get_mean_curve(self):
        """
        求过去N帧曲率的均值
        """
        n = len(self.detection_results)
        m = len(self.detection_results[0].curvs)
        result = [0] * m
        for i in range(m):
            for j in range(n):
                result[i] += self.detection_results[j].curvs[i]
            result[i] /= n
        return result

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

        # if fit_success:
        #     print('fit line success')

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
        # fit_success,left_distance,right_distance,curvatures,fit_on_front_img = utils.post_process_on_bev(result_binary_img,self.current_frame,self.M,self.M_inv)
        fit_success,left_fit,right_fit,leftx,rightx,out_img = utils.post_process_on_bev(result_binary_img,self.current_frame,self.M,self.M_inv)
        t = (time.time() - t0)*1000
        if PRINT_TIME_STATIC:
            print('utils.post_process_on_bev time: {} ms!'.format(t))

        if fit_success:
            #计算距离左右车道线的距离
            left_distance,right_distance = utils.cal_distance(utils.CAR_PIXEL_ON_FRONT,self.M,left_fit,right_fit)
            print('left_distance:{},right_distance:{}'.format(left_distance,right_distance))

            #计算曲率. 使用左车道线拟合曲线
            curvatures = utils.cal_curvature(left_fit)
            print('curvatures:{}'.format(curvatures))

            #计算此前的曲率均值
            previous_stabel_curves=None
            if len(self.detection_results) > 0:
                previous_stabel_curves = self.get_mean_curve()
                
            #保存n帧检测结果
            current_frame_result = LaneDetectionResults() 
            current_frame_result.left_fit = left_fit
            current_frame_result.right_fit = right_fit
            current_frame_result.curvs = curvatures
            # print('current_frame_result:*********************')
            # current_frame_result.print_details()

            if len(self.detection_results) >= MAX_FRAMES_NUM:
                self.detection_results.pop(0)
            self.detection_results.append(current_frame_result)

            #判断当前帧的检测结果是否可用
            fit_on_front_img = None
            curve_change_too_much = False
            if previous_stabel_curves:
                print('previous_stabel_curves is {}'.format(previous_stabel_curves))
                curve_change = [a / b for a, b in zip(previous_stabel_curves, current_frame_result.curvs)]
                if any(abs(e) > (1 + CURVE_CHANGE_THRESHOLD) for e in curve_change):
                    print('curve change too big')
                    curve_change_too_much = True

            #确定要使用的曲率方程
            prefer_left_fit,prefer_right_fit = None,None
            if curve_change_too_much and REFER_PAST_FRAMES:
                prefer_left_fit =  self.prefer_detection_result.left_fit
                prefer_right_fit = self.prefer_detection_result.right_fit
                print('use previous curve,left:{},right:{}'.format(prefer_left_fit,prefer_right_fit))
                self.prefer_detection_result.print_details()
            else:
                prefer_left_fit =  left_fit
                prefer_right_fit = right_fit

                self.prefer_detection_result = current_frame_result
                print('use current curve,left:{},right:{}'.format(prefer_left_fit,prefer_right_fit))
                self.prefer_detection_result.print_details()

            #绘制了车道线曲线的透视图变换到原图视角
            t0 = time.time()
            if utils.NEED_VISUALIZATION:
                # bev图上根据拟合的曲线方程绘制车道线
                h,w = self.current_frame.shape[0],self.current_frame.shape[1]
                ploty = np.linspace(int(h/2), h-1, int(h/2))
                left_fitx = prefer_left_fit[0]*ploty**2 + prefer_left_fit[1]*ploty + prefer_left_fit[2]
                right_fitx = prefer_right_fit[0]*ploty**2 + prefer_right_fit[1]*ploty + prefer_right_fit[2]
                left_fitx = [int(e) for e in left_fitx]
                right_fitx = [int(e) for e in right_fitx]

                ploty = [int(e) for e in ploty ]
                for i in range(-5,5): #只画一条线的话　太模糊了　看不出来
                    draw_left_fitx = [min(w-1,max(0,e+i)) for e in left_fitx]
                    draw_right_fitx = [min(e+i,w-1) for e in right_fitx]
                    out_img[ploty,draw_left_fitx] = [255,0,0]
                    out_img[ploty,draw_right_fitx] = [255,0,0]

                line_img = utils.warper(out_img, self.M_inv)
                print('convert from bev to front')
                fit_on_front_img = cv2.addWeighted(self.current_frame, 1, line_img, 2, 0)
                description = 'left:{:.2f},right:{:.2f}'.format(left_distance,right_distance)
                cv2.putText(fit_on_front_img, description, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # cv2.imwrite('prediction_linefit_on_origin_img.png',fit_on_front_img)
            t = (time.time() - t0)*1000
            if PRINT_TIME_STATIC:
                print('VISUALIZATION time: {} ms!'.format(t))

            if FOR_SEG_VISULIZATION:
                print('result_binary_img shape:{}'.format(result_binary_img.shape))
                seg_on_front_img = self.current_frame
                lane_mask = (result_binary_img==255)
                seg_on_front_img[lane_mask] = (255, 0, 0)
                print('type(lane_mask):{},shape:{}'.format(type(lane_mask),lane_mask.shape))
                fit_on_front_img = seg_on_front_img

            return fit_success,left_distance,right_distance,curvatures,fit_on_front_img
        else:
            return fit_success,None,None,None,None
    

def test_on_img(img_path):
    # lane_node = lane_detect_node('./runs/train/banqiao_cam8m5/weights/epoch_299.pt',model_input_size=(544,960))
    lane_node = lane_detect_node('./runs/train/banqiao_cam8M_1280x7362/weights/epoch_239.pt',model_input_size=(736,1280))
    img = cv2.imread(img_path)
    fit_success,left_distance,right_distance,curvatures,fit_on_front_img = lane_node.detect(img)
    cv2.imwrite('./fit_on_front_img.png',fit_on_front_img)

def export2onnx(pt_path,onnx_save_path,input_size):
    export.export_onnx(pt_path,onnx_save_path,input_size)

def main(args=None):
    # lane_node = lane_detect_node('./runs/train/banqiao_cam8m4/weights/epoch_299.pt')
    # lane_node = lane_detect_node('./runs/train/banqiao_cam8m5/weights/epoch_299.pt',model_input_size=(544,960))
    lane_node = lane_detect_node('./runs/train/banqiao_cam8M_1280x7362/weights/epoch_239.pt',model_input_size=(736,1280))

    filelist=[]
    datadir = '/home/autocore/work_sc/datasets/banqiao/20230418/road03/Image'
    videoname = 'banqiao_cam8m_road03.avi'
    utils.get_file(datadir,filelist,sortKey=utils.sort_by_time)
    # print(filelist)

    fit_on_front_img_list = []
    video_size = None
    for i,img_path in enumerate(filelist):
        img = cv2.imread(img_path)
        fit_success,left_distance,right_distance,curvatures,fit_on_front_img = lane_node.detect(img)
        if fit_success:
            fit_on_front_img_list.append(fit_on_front_img)

            height, width, _ = fit_on_front_img.shape
            video_size = (width,height)

            # print('img_path:{}'.format(img_path))
            details = img_path.split('/')
            # print(details)

            save_dir = '/'.join(details[:-2]) + '/Predict'
            # print('save_dir is :{}'.format(save_dir))

            save_name = img_path.split('/')[-1].replace('.png','_bin.png')
            # print('save_name is :{}'.format(save_name))

            save_full_path = '{}/{}'.format(save_dir,save_name)
            # print('save_full_path is {}'.format(save_full_path))

            cv2.imwrite(save_full_path,fit_on_front_img)

    print('saving {} to video:{}'.format(len(fit_on_front_img_list),videoname))
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), 15, video_size)
    for i in range(len(fit_on_front_img_list)):
        out.write(fit_on_front_img_list[i])
        
    out.release()


if __name__ == '__main__':
    # test_on_img('/home/autocore/work_sc/datasets/banqiao/20230418/road02/Image/1681827555_642832180.png')
    # test_on_img('/home/autocore/work_sc/yolov7/test.png')
    # export2onnx('./runs/train/banqiao_cam8m5/weights/epoch_299.pt','banqiao_8m_544x960.onnx')
    export2onnx('./runs/train/banqiao_cam8M_1280x7362/weights/epoch_239.pt','banqiao_8m_736x1280_newname.onnx',input_size=(736,1280))
    # main()
