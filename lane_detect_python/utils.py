from ast import If
from distutils.debug import DEBUG
from pickle import TRUE
import cv2
import numpy as np
import os 
import time

PRINT_TIME_STATIC=True

#8M相机 3840x2160
# CAR_PIXEL_ON_FRONT = (1649,2157) #前视角图上的自车位置.
# M_EVERY_PIXEL_ON_BEV = 0.1 #BEV图上的每个像素对应的距离,单位:米
# Y_FOR_CAL_CURVATURE = [1000,1500,2000] #bev图上用来计算曲率的像素位置的y值.
# LANE_PIXEL_LEN = 340 #透视图上一个车道宽度的像素个数　实际测出来

# #8M相机 960x540
# CAR_PIXEL_ON_FRONT = (480,540) #前视角图上的自车位置.
# M_EVERY_PIXEL_ON_BEV = 0.1 #BEV图上的每个像素对应的距离,单位:米
# Y_FOR_CAL_CURVATURE = [300,400,500] #bev图上用来计算曲率的像素位置的y值.
# LANE_PIXEL_LEN = 34 #透视图上一个车道宽度的像素个数　实际测出来
# LANE_THRESHOLD = 1.2 * LANE_PIXEL_LEN #控制bev图上　x方向上的搜索范围
# WINDOW_SIZE_X = LANE_PIXEL_LEN #x方向滑窗的大小


# #8M相机 1280x736
# NEW_SIZE=(1280,736)
# x = [468, 402, 778,722]
# y = [462, 515, 515,461]
# X = [400, 400, 550, 550] 
# Y = [550, 600, 600, 550]
# X_offset,Y_offset = 150,200 #手动调整找到合适的点
# X = [e + X_offset for e in X] #
# Y = [e + Y_offset for e in Y]
# CAR_PIXEL_ON_FRONT = (640,736) #前视角图上的自车位置.
# Y_FOR_CAL_CURVATURE = [300,400,500] #bev图上用来计算曲率的像素位置的y值.
# LANE_PIXEL_LEN = X[2]-X[1] #透视图上一个车道宽度的像素个数
# M_EVERY_PIXEL_ON_BEV = 3.5/LANE_PIXEL_LEN #BEV图上的每个像素对应的距离,单位:米
# LANE_THRESHOLD = 1.2 * LANE_PIXEL_LEN #控制bev图上　x方向上的搜索范围
# WINDOW_SIZE_X = int(LANE_PIXEL_LEN/2) #x方向滑窗的大小

#20230426 1280x736
NEW_SIZE=(1280,736)
x = [468, 402, 778,722]
y = [462, 515, 515,461]
X = [400, 400, 550, 550] 
Y = [500, 550, 550, 500]
X_offset,Y_offset = 150,100 #手动调整找到合适的点
X = [e + X_offset for e in X] #
Y = [e + Y_offset for e in Y]
CAR_PIXEL_ON_FRONT = (640,736) #前视角图上的自车位置.
Y_FOR_CAL_CURVATURE = [300,400,500] #bev图上用来计算曲率的像素位置的y值.
LANE_PIXEL_LEN = X[2]-X[1] #透视图上一个车道宽度的像素个数
M_EVERY_PIXEL_ON_BEV = 3.5/LANE_PIXEL_LEN #BEV图上的每个像素对应的距离,单位:米
LANE_THRESHOLD = 1.2 * LANE_PIXEL_LEN #控制bev图上　x方向上的搜索范围
WINDOW_SIZE_X = int(LANE_PIXEL_LEN/2) #x方向滑窗的大小

# 原2M相机
# CAR_PIXEL_ON_FRONT = (2285,2697) #前视角图上的自车位置.
# M_EVERY_PIXEL_ON_BEV = 0.1 #BEV图上的每个像素对应的距离,单位:米
# Y_FOR_CAL_CURVATURE = [300,500,700] #bev图上用来计算曲率的像素位置的y值.
# LANE_PIXEL_LEN = 300 #透视图上一个车道宽度的像素个数　实际测出来

NEED_VISUALIZATION=TRUE #是否将拟合的曲线绘制到原图


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # print('img.dtype:{},sum:{}'.format(img.dtype,np.sum(img)))

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
    # print('ratio:{}'.format(ratio))
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # print('dw:{},dh:{}'.format(dw,dh))
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # print('dw:{},dh:{}'.format(dw,dh))

    if shape[::-1] != new_unpad:  # resize
        print('resize******************************')
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # print('top:{},bottom:{},left:{},right:{}'.format(top,bottom,left,right))
    # print('img shape:{}'.format(img.shape))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # print('img shape:{}'.format(img.shape))

    # print('after letterbox,sum:{}'.format(np.sum(img)))
    return img, ratio, (dw, dh)

def warper(img, M):
    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def fit_line(binary_bev_img,visualization=True):
    print('fit_line***********************')
    cv2.imwrite('search_on_this_binary_bev_img.png',binary_bev_img)
    #确定滑窗开始的起始位置
    h,w = binary_bev_img.shape
    histogram = np.sum(binary_bev_img[int(h/2):,:], axis=0) #图像下半部分每一列的非0像素个数
    #print('histogram:{}'.format(histogram.shape))
    midpoint = np.int32(histogram.shape[0]/2)#宽度的一半
    
    start = int(midpoint - LANE_THRESHOLD)
    end = int(midpoint + LANE_THRESHOLD)
    print('start:{},midpoint:{},end:{}'.format(start,midpoint,end))
    leftx_base = np.argmax(histogram[start : midpoint]) + start
    rightx_base = np.argmax(histogram[midpoint:end]) + midpoint
    leftx_current = leftx_base
    rightx_current = rightx_base
    print('leftx_base:{},rightx_base:{}'.format(leftx_base,rightx_base))
    
    #确定所有车道线点的下标
    lane_pixel = binary_bev_img.nonzero()
    lane_pixel_y = np.array(lane_pixel[0])
    lane_pixel_x = np.array(lane_pixel[1])
    #print('lane_pixel_y:{},lane_pixel_x:{}'.format(lane_pixel_y.shape,lane_pixel_x.shape))
    #存储左右车道线点下标
    left_lane_inds = []
    right_lane_inds = []
    
    #设定滑窗的大小
    nwindows = 9
    window_size_x,window_size_y = WINDOW_SIZE_X,int(binary_bev_img.shape[0]/nwindows)
    all_windows=[]
    for window in range(nwindows):
        #确定当前滑窗范围　
        win_xleft_low = leftx_current - window_size_x
        win_xleft_high = leftx_current + window_size_x

        win_xright_low = max(0,rightx_current - window_size_x)
        win_xright_high = min(w-1,rightx_current + window_size_x)
        
        win_y_low = max(0,binary_bev_img.shape[0] - (window+1)*window_size_y)
        win_y_high = min(h-1,binary_bev_img.shape[0] - window*window_size_y)
        
        left_window = ((win_xleft_low,win_y_low),(win_xleft_high,win_y_high))
        right_window = ((win_xright_low,win_y_low),(win_xright_high,win_y_high))
        all_windows.append((left_window,right_window))
        # print('left_window:{},right_window:{}'.format(left_window,right_window))

        #确定滑窗内的车道线点
        good_left_inds = ((lane_pixel_y >= win_y_low) & (lane_pixel_y < win_y_high) & (lane_pixel_x >= win_xleft_low) & (lane_pixel_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((lane_pixel_y >= win_y_low) & (lane_pixel_y < win_y_high) & (lane_pixel_x >= win_xright_low) & (lane_pixel_x < win_xright_high)).nonzero()[0]
        #print('good_left_inds:{}'.format(good_left_inds.shape))
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
         
        #更新下一次搜索的位置
        minpix = 5 #滑窗内的最小车道线点个数
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(lane_pixel_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(lane_pixel_x[good_right_inds]))

    #绘制出滑窗搜索的过程
    print('draw search process*******************')
    search_img = np.dstack((binary_bev_img,binary_bev_img,binary_bev_img))
    thickness = 2
    search_img = cv2.line(search_img,(midpoint,0),(midpoint,h-1),(0,0,255),thickness)
    search_img = cv2.line(search_img,(start,0),(start,h-1),(0,255,0),thickness)
    search_img = cv2.line(search_img,(end,0),(end,h-1),(0,255,0),thickness)
    for window in all_windows:
        left_window,right_window = window[0],window[1]
        # print('left_window:{},right_window:{}'.format(left_window,right_window))
        left_color = (255, 0, 0)
        right_color = (197,145,99)
        search_img = cv2.rectangle(search_img, left_window[0], left_window[1], left_color, thickness)
        search_img = cv2.rectangle(search_img, right_window[0], right_window[1], right_color, thickness)
    cv2.imwrite('prediction_binary_search_windows.png',search_img)
    print('draw prediction_binary_search_windows end')

    #提取左右车道线点的像素坐标
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = lane_pixel_x[left_lane_inds]
    lefty = lane_pixel_y[left_lane_inds]
    rightx = lane_pixel_x[right_lane_inds]
    righty = lane_pixel_y[right_lane_inds]

    #做二次多项式拟合
    # print('lefty:{},leftx:{}'.format(lefty,leftx))
    left_fit = np.polyfit(lefty, leftx, 2)
    # print('righty:{},rightx:{}'.format(righty,rightx))
    right_fit = np.polyfit(righty, rightx, 2)
    
    #绘制曲线
    if visualization:
        out_img = np.dstack((binary_bev_img,binary_bev_img,binary_bev_img)) * 114
        out_img = out_img.astype('uint8')

        #根据曲线方程算出每一个y对应的x  
        ploty = np.linspace(int(h/2), h-1, int(h/2)) 
        #print('h is :{}'.format(h))
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        #车道线像素点绘制
        out_img[lane_pixel_y[left_lane_inds], lane_pixel_x[left_lane_inds]] = [0,0,255]
        out_img[lane_pixel_y[right_lane_inds], lane_pixel_x[right_lane_inds]] = [0, 0, 255]

        #车道线拟合曲线绘制
        ploty = [int(e) for e in ploty ]
        # print('ploty:{}'.format(ploty))
        left_fitx = [int(e) for e in left_fitx]
        right_fitx = [int(e) for e in right_fitx]
        for i in range(-5,5): #只画一条线的话　太模糊了　看不出来
            draw_left_fitx = [max(0,e+i) for e in left_fitx]
            draw_right_fitx = [min(e+i,w) for e in right_fitx]
            out_img[ploty,draw_left_fitx] = [255,0,0]
            out_img[ploty,draw_right_fitx] = [255,0,0]

        cv2.imwrite('prediction_linefit_onbev.png',out_img)
    print('draw line on bev end')

    return left_fit,right_fit,leftx,rightx,out_img

def cal_curvature(fit):
    """
    bev图上:left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    """
    #print('fit:{}'.format(fit))
    d = np.polyder(fit) #导数

    curvature = np.polyval(d, Y_FOR_CAL_CURVATURE)

    return curvature
    
def cal_distance(self_car_pixel,M,left_fit,right_fit):
    car_vec = np.array([self_car_pixel[0],self_car_pixel[1],1])
    car_vec_in_bird_view = M.dot(car_vec.T)
    car_vec_in_bird_view /= car_vec_in_bird_view[2]
    car_x_bev,car_y_bev = car_vec_in_bird_view[0],car_vec_in_bird_view[1]
    left_x = left_fit[0]*car_y_bev**2 + left_fit[1]*car_y_bev + left_fit[2]
    left_pixel_distance = car_x_bev - left_x
    right_x = right_fit[0]*car_y_bev**2 + right_fit[1]*car_y_bev + right_fit[2]
    right_pixel_distance = right_x - car_x_bev
    
    print('cal_distance,car_x_bev:{},left_x:{},right_x:{}'.format(car_x_bev,left_x,right_x))

    left_distance = M_EVERY_PIXEL_ON_BEV * left_pixel_distance
    right_distance = M_EVERY_PIXEL_ON_BEV * right_pixel_distance

    return left_distance,right_distance

def post_process_on_bev(prediction_binary_img,origin_img,M,M_inv):    
    fit_on_front_img = origin_img
    fit_success = False 
    left_distance,right_distance,curvatures = -1.0,-1.0,[-1.0,-1.0,-1.0]
    
    print('post_process_on_bev*********************************')
    #这里prediction_binary_img是交给前处理之前的图片.
    print('prediction_binary_img:{}'.format(prediction_binary_img.shape))
    cv2.imwrite('prediction_binary_on_origin.png',prediction_binary_img) #在原图尺寸上的分割图
    
    # print('M is :{}'.format(M))
    t0 = time.time()
    binary_bev_img = warper(prediction_binary_img,M)
    t = (time.time() - t0)*1000
    if PRINT_TIME_STATIC:
            print('from front img to bev img time: {} ms!'.format(t))
    
    cv2.imwrite('prediction_binary_on_bev.png',binary_bev_img)
    try:
        t0 = time.time()
        left_fit,right_fit,leftx,rightx,out_img = fit_line(binary_bev_img)
        t = (time.time() - t0)*1000
        if PRINT_TIME_STATIC:
            print('fit_line time: {} ms!'.format(t))
        
        print('draw line on bev end********')

        #计算距离左右车道线的距离
        left_distance,right_distance = cal_distance(CAR_PIXEL_ON_FRONT,M,left_fit,right_fit)
        print('left_distance:{},right_distance:{}'.format(left_distance,right_distance))

        #计算曲率. 使用左车道线拟合曲线
        curvatures = cal_curvature(left_fit)
        print('curvatures:{}'.format(curvatures))


        ##绘制了车道线曲线的透视图变换到原图视角
        t0 = time.time()
        if NEED_VISUALIZATION:
            line_img = warper(out_img, M_inv)
            # line_img = cv2.warpPerspective(out_img, M_inv, (out_img.shape[1], out_img.shape[0]), flags=cv2.INTER_NEAREST) 
            print('convert from bev to front')
            # fit_on_front_img = cv2.addWeighted(origin_img, 1, prediction_binary_img, 2, 0)
            fit_on_front_img = cv2.addWeighted(origin_img, 1, line_img, 2, 0)
            description = 'left:{:.2f},right:{:.2f}'.format(left_distance,right_distance)
            cv2.putText(fit_on_front_img, description, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.imwrite('prediction_linefit_on_origin_img.png',fit_on_front_img)
        t = (time.time() - t0)*1000
        if PRINT_TIME_STATIC:
            print('VISUALIZATION time: {} ms!'.format(t))
        
        fit_success = True
    except:
        print('eeeeeeeeeeeeeeeeeeeee')
    
    return fit_success,left_distance,right_distance,curvatures,fit_on_front_img


def get_file(dirname,filelist=[],sortKey=None):
    for dirpath, dirname, filenames in os.walk(dirname):
        # if sort:
        #     filenames = sorted(filenames) #要排序　否则会乱序
        for filename in sorted(filenames,key=sortKey): 
            if filename.endswith('jpg') or filename.endswith('png'):
                fullpath = os.path.join(dirpath, filename)
                filelist.append(fullpath)


def sort_by_time(file_name):
    """
    文件名是数字｀
    """
    file_name_num = file_name[:-4]
    nums = file_name_num.split('_')
   
    return (int(nums[0]),int(nums[1]))
    

import onnx
import torch
def export_onnx(pt_path,onnx_save_path):
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
   
    output_names = ['output']
    
    # 加载模型 不清楚原因,只能在cpu上处理 torch版本:1.13.1+cu117
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = weigths['model'].float() #模型转换为fp32
    model = model.to(device)
    _ = model.eval()
 
    # Input
    img = torch.zeros(1, 3, 960,960) # image size(1,3,320,192) iDetection
    # img = img.half().to(device)
    img = img.to(device)
    print('img shape:{},is_cuda:{}'.format(img.shape,img.is_cuda))

    print('prepare to export*******')
    torch.onnx.export(model, img, onnx_save_path, verbose=False, opset_version=11, input_names=['images'],
                          output_names=output_names)
    print('export done*******')

    # Checks
    onnx_model = onnx.load(onnx_save_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
