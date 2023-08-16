from ast import If, Return
from distutils.debug import DEBUG
from pickle import FALSE, TRUE
import cv2
import numpy as np
import os 
import time
import math

PRINT_TIME_STATIC=True
NEED_VISUALIZATION=True #是否将拟合的曲线绘制到原图
SAVE_DEBUG_PNG = True 
CURVE_THETA=True #是否需要将切线斜率转换成角度

#20230426 8M相机 1280x736
# x = [468, 402, 778,722]
# y = [462, 515, 515,461]
# X = [400, 400, 550, 550] 
# Y = [500, 550, 550, 500]
# X_offset,Y_offset = 150,100 #手动调整找到合适的点
# X = [e + X_offset for e in X] #
# Y = [e + Y_offset for e in Y]
# src = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]))
# dst = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]],[X[2], Y[2]], [X[3], Y[3]]]))
# self.M = cv2.getPerspectiveTransform(src, dst)
# self.M_inv = cv2.getPerspectiveTransform(dst, src)


NEW_SIZE=(1280,736)
H = np.array([[-0.1659823316008509, -1.87100288155966, 739.6450148427723],
[4.402169850633356e-09, -2.205073584926861, 816.4599842909245],
[8.890583548179584e-12, -0.002954215096412131, 1]],dtype=np.float)

H_inv = np.array([[-6.024737554046832, 9.144261815093722, -3009.756760316751],
[-8.317262657058759e-08, 4.832673024743682, -3945.684080347182],
[-1.921463964429701e-10, 0.01427675552442364, -10.65639944907617]],dtype=np.float)

CAR_PIXEL_ON_FRONT = (640,736) #前视角图上的自车位置.
Y_FOR_CAL_CURVATURE = [300,400,500] #bev图上用来计算曲率的像素位置的y值.
LANE_PIXEL_LEN = 150 #透视图上一个车道宽度的像素个数
M_EVERY_PIXEL_ON_BEV = 3.5/LANE_PIXEL_LEN #BEV图上的每个像素对应的距离,单位:米

#fit_line相关控制参数
# LANE_THRESHOLD = 1.2 * LANE_PIXEL_LEN #控制bev图上　x方向上的搜索范围
LANE_THRESHOLD = 0.6 * LANE_PIXEL_LEN #控制bev图上　x方向上的搜索范围
WINDOW_SIZE_X = int(LANE_PIXEL_LEN/3.) #x方向滑窗的大小
FIT_LINE_H_START = int(NEW_SIZE[1] * 2./3)
FIT_LINE_H_END = int(NEW_SIZE[1] * 95./100)
CAM_CENTER_TO_LEFT = 1.5 #相机中心到车辆左侧边沿距离
CAM_CENTER_TO_RIGHT = 0.3 #相机中心到车辆右侧边沿距离

pre_search_left_x = None
pre_search_right_x = None

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

def warp_imwrite(img_path,img):
    if SAVE_DEBUG_PNG:
        print('save img*********************************')
        cv2.imwrite(img_path,img)

def decide_serach_start_point(line_fit):
    """
    根据曲线方程line_fit推断出当前帧的搜索起点位置
    line_fit暂时取上一帧的左车道线的方程
    """
    leftx_base,rightx_base = 0,0
    y_bev = NEW_SIZE[1]
    leftx_base = line_fit[0]*y_bev**2 + line_fit[1]*y_bev + line_fit[2]
    rightx_base = leftx_base + LANE_PIXEL_LEN

    return leftx_base,rightx_base

def cal_search_base(pre_line_fit,img):
    """
    根据上一帧的曲线方程确定当前帧的搜索起点
    """
    k = 1 #　x = ky + b
    for b in range(int(-LANE_THRESHOLD/2),int(LANE_THRESHOLD/2)):
        lane_pixel_num = 0
        for y in range(FIT_LINE_H_START,h):
            x = ky + b
            if (255 == img[y,x]):
                lane_pixel_num +=1

def get_search_base(binary_bev_img):
    """
    """     
    if (pre_search_left_x is None) and (pre_search_right_x is None):
        h,w = binary_bev_img.shape
        histogram = np.sum(binary_bev_img[FIT_LINE_H_START:,:], axis=0) #图像下半部分每一列的非0像素个数  adjust this to avoid rotation
        #print('histogram:{}'.format(histogram.shape))
        midpoint = np.int32(histogram.shape[0]/2)#宽度的一半
        
        start = int(midpoint - LANE_THRESHOLD)
        end = int(midpoint + LANE_THRESHOLD) #
        print('start:{},midpoint:{},end:{}'.format(start,midpoint,end))
        leftx_base = np.argmax(histogram[start : midpoint]) + start
        rightx_base = np.argmax(histogram[midpoint:end]) + midpoint
        leftx_current = leftx_base
        rightx_current = rightx_base
        print('leftx_base:{},rightx_base:{}'.format(leftx_base,rightx_base))   

        return leftx_current,rightx_current
    else:
        return pre_search_left_x,pre_search_right_x

def fit_line(binary_bev_img,visualization=True):
    print('fit_line***********************')
    warp_imwrite('search_on_this_binary_bev_img.png',binary_bev_img)
    #确定滑窗开始的起始位置
    h,w = binary_bev_img.shape
    histogram = np.sum(binary_bev_img[FIT_LINE_H_START:,:], axis=0) #图像下半部分每一列的非0像素个数  adjust this to avoid rotation
    #print('histogram:{}'.format(histogram.shape))
    midpoint = np.int32(histogram.shape[0]/2)#宽度的一半
    
    start = int(midpoint - LANE_THRESHOLD)
    end = int(midpoint + LANE_THRESHOLD) #
    print('start:{},midpoint:{},end:{}'.format(start,midpoint,end))
    leftx_base = np.argmax(histogram[start : midpoint]) + start
    rightx_base = np.argmax(histogram[midpoint:end]) + midpoint
    leftx_current = leftx_base
    rightx_current = rightx_base
    print('leftx_base:{},rightx_base:{}'.format(leftx_base,rightx_base))
    
    # leftx_current,rightx_current = get_search_base(binary_bev_img)

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
    print('x方向滑动大小:{},y方向滑动大小:{}'.format(window_size_x,window_size_y))
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
        print('left_window:{},right_window:{}'.format(left_window,right_window))

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
    warp_imwrite('prediction_binary_search_windows.png',search_img)
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

        warp_imwrite('prediction_linefit_onbev.png',out_img)
    print('draw line on bev end')

    return left_fit,right_fit,leftx,rightx,out_img

def cal_curvature(fit):
    """
    bev图上:left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    """
    #print('fit:{}'.format(fit))
    d = np.polyder(fit) #导数

    curvature = np.polyval(d, Y_FOR_CAL_CURVATURE)
    
    if CURVE_THETA:
        return [math.degrees(math.atan(e)) for e in curvature]
    else:
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

def ajust_distance(left_distance,right_distance,alpha):
    """
    将相机中心距离左右车道的距离转换成车辆左右边沿距离车道线的距离
    由于车身在车道内并非完全平行于车道线,bev图的底边中点实际上是摄像头在地面垂线点沿着摄像头方向4m左右的地面点
    """
    left_distance = left_distance - 4.0 * math.sin(alpha) - CAM_CENTER_TO_LEFT 
    right_distance = right_distance + 4.0 * math.sin(alpha) - CAM_CENTER_TO_RIGHT 

    return left_distance,right_distance

def post_process_on_bev(prediction_binary_img,origin_img,M,M_inv):    
    fit_on_front_img = origin_img
    fit_success = False 
    left_distance,right_distance,curvatures = -1.0,-1.0,[-1.0,-1.0,-1.0]
    
    print('post_process_on_bev*********************************')
    #这里prediction_binary_img是交给前处理之前的图片.
    print('prediction_binary_img:{}'.format(prediction_binary_img.shape))
    warp_imwrite('prediction_binary_on_origin.png',prediction_binary_img) #在原图尺寸上的分割图
    
    # print('M is :{}'.format(M))
    t0 = time.time()
    binary_bev_img = warper(prediction_binary_img,M)
    t = (time.time() - t0)*1000
    if PRINT_TIME_STATIC:
            print('from front img to bev img time: {} ms!'.format(t))
    
    warp_imwrite('prediction_binary_on_bev.png',binary_bev_img)
    try:
        t0 = time.time()
        left_fit,right_fit,leftx,rightx,out_img = fit_line(binary_bev_img)
        t = (time.time() - t0)*1000
        if PRINT_TIME_STATIC:
            print('fit_line time: {} ms!'.format(t))
        
        print('draw line on bev end********')

        # #计算距离左右车道线的距离
        # left_distance,right_distance = cal_distance(CAR_PIXEL_ON_FRONT,M,left_fit,right_fit)
        # print('left_distance:{},right_distance:{}'.format(left_distance,right_distance))

        # #计算曲率. 使用左车道线拟合曲线
        # curvatures = cal_curvature(left_fit)
        # print('curvatures:{}'.format(curvatures))


        # ##绘制了车道线曲线的透视图变换到原图视角
        # t0 = time.time()
        # if NEED_VISUALIZATION:
        #     line_img = warper(out_img, M_inv)
        #     # line_img = cv2.warpPerspective(out_img, M_inv, (out_img.shape[1], out_img.shape[0]), flags=cv2.INTER_NEAREST) 
        #     print('convert from bev to front')
        #     # fit_on_front_img = cv2.addWeighted(origin_img, 1, prediction_binary_img, 2, 0)
        #     fit_on_front_img = cv2.addWeighted(origin_img, 1, line_img, 2, 0)
        #     description = 'left:{:.2f},right:{:.2f}'.format(left_distance,right_distance)
        #     cv2.putText(fit_on_front_img, description, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #     warp_imwrite('prediction_linefit_on_origin_img.png',fit_on_front_img)
        # t = (time.time() - t0)*1000
        # if PRINT_TIME_STATIC:
        #     print('VISUALIZATION time: {} ms!'.format(t))
        
        fit_success = True
    except:
        print('eeeeeeeeeeeeeeeeeeeee')
    
    # return fit_success,left_distance,right_distance,curvatures,fit_on_front_img
    if fit_success:
        return fit_success,left_fit,right_fit,leftx,rightx,out_img
    else:
        return fit_success,None,None,None,None,None

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
    
def test_fit_line():
    test_png = '/home/sc/work/dlDev/mivii_orin/lane_detect_ros_python/images/20230509/prediction_binary_on_origin.png'
    binary_bev_img = warper(cv2.imread(test_png,cv2.IMREAD_GRAYSCALE), H)
    # binary_bev_img = cv2.imread(test_png,cv2.IMREAD_GRAYSCALE)  
    fit_line(binary_bev_img,visualization=True)

if __name__ == '__main__':
    test_fit_line()