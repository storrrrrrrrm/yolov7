import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
import os
import time

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
    result_on_origin_img = img.copy()#在原图上绘制
    result_binary_img = np.zeros( (img.shape[0],img.shape[1]) )
    img, ratio, (pad_w,pad_h) = letterbox(img,1280,auto=False, scaleup=False) 
    print('ratio:{},pad_w:{},pad_h:{}'.format(ratio,pad_w,pad_h))
    result_img = np.zeros_like(img) #letterbox
    img = img[:, :, ::-1].transpose(2, 0, 1)  # rgb chw
    img = np.ascontiguousarray(img)
    img = img/255.

    img = torch.from_numpy(img)
    img = torch.unsqueeze(img,0)
    img = img.half().to(device)

    start=time.time()
    output = model(img)
    end=time.time()
    print('forward time:{}'.format(end-start))
    _,lane_pre = output[0],output[1] #lane 2x640x640
    lane_pre = lane_pre.float().cpu() #bchw

    
    b = lane_pre.shape[0]
    for i in range(b):  
        current_lane_pre = torch.sigmoid(lane_pre[i,...])
        current_lane_pre_mask = np.where(current_lane_pre>0.7)
        pre_lane_num = len(current_lane_pre_mask[1])
        print('当前检测出车道点个数:{}'.format(pre_lane_num))
        # print(current_lane_pre_mask[1])
        # print(current_lane_pre_mask[2])

        #模型输出img
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

                result_binary_img[h,w] = result_img[new_h,new_w,2]

        return result_on_origin_img,result_binary_img

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
    for i,img_path in enumerate(filelist):
        start = time.time()
        result_on_origin_img,result_binary_img = predict(img_path)
        end = time.time()
        print('predict on {} image:{},time:{:.2f}'.format(i,img_path,end-start))

        origin_img = cv2.imread(img_path)
        result_img = post_process(result_binary_img,origin_img)
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

    print('saving {} to video:{}'.format(len(result_img_list),videoname))
    print('size:{}'.format(size))
    out = cv2.VideoWriter(videoname,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(result_img_list)):
        # print('{}'.format(result_img_list[i].shape))
        out.write(result_img_list[i])
        # print('save {}'.format(filelist[i]))
        
    out.release()

def fit_line(binary_bev_img,visualization=True):
    #确定滑窗开始的起始位置
    h,w = binary_bev_img.shape
    histogram = np.sum(binary_bev_img[int(h/2):,:], axis=0) #图像下半部分每一列的非0像素个数
    print('histogram:{}'.format(histogram.shape))
    midpoint = np.int32(histogram.shape[0]/2)#宽度的一半
    LANE_PIXEL_LEN = 300 #透视图上一个车道宽度的像素个数　实际测出来
    LANE_THRESHOLD = 1.5 * LANE_PIXEL_LEN #控制x方向上的搜索范围
    start = int(midpoint - LANE_THRESHOLD)
    end = int(midpoint + LANE_THRESHOLD)
    leftx_base = np.argmax(histogram[start : midpoint]) + start
    rightx_base = np.argmax(histogram[midpoint:end]) + midpoint
    leftx_current = leftx_base
    rightx_current = rightx_base
    print('leftx_base:{},rightx_base:{}'.format(leftx_base,rightx_base))
    
    #确定所有车道线点的下标
    lane_pixel = binary_bev_img.nonzero()
    lane_pixel_y = np.array(lane_pixel[0])
    lane_pixel_x = np.array(lane_pixel[1])
    print('lane_pixel_y:{},lane_pixel_x:{}'.format(lane_pixel_y.shape,lane_pixel_x.shape))
    #存储左右车道线点下标
    left_lane_inds = []
    right_lane_inds = []

    
    #设定滑窗的大小
    nwindows = 9
    window_size_x,window_size_y = 100,int(binary_bev_img.shape[0]/nwindows)
    all_windows=[]
    for window in range(nwindows):
        #确定当前滑窗范围　
        win_xleft_low = leftx_current - window_size_x
        win_xleft_high = leftx_current + window_size_x

        win_xright_low = rightx_current - window_size_x
        win_xright_high = rightx_current + window_size_x
        
        win_y_low = binary_bev_img.shape[0] - (window+1)*window_size_y
        win_y_high = binary_bev_img.shape[0] - window*window_size_y
        
        left_window = ((win_xleft_low,win_y_low),(win_xleft_high,win_y_high))
        right_window = ((win_xright_low,win_y_low),(win_xright_high,win_y_high))
        all_windows.append((left_window,right_window))
        
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
    search_img = np.dstack((binary_bev_img,binary_bev_img,binary_bev_img)) * 255
    thickness = 2
    search_img = cv2.line(search_img,(midpoint,0),(midpoint,h-1),(0,0,255),thickness)
    search_img = cv2.line(search_img,(start,0),(start,h-1),(0,255,0),thickness)
    search_img = cv2.line(search_img,(end,0),(end,h-1),(0,255,0),thickness)
    for window in all_windows:
        left_window,right_window = window[0],window[1]
        color = (255, 0, 0)
        search_img = cv2.rectangle(search_img, left_window[0], left_window[1], color, thickness)
        search_img = cv2.rectangle(search_img, right_window[0], right_window[1], color, thickness)
    cv2.imwrite('prediction_binary_search_windows.png',search_img)

    #提取左右车道线点的像素坐标
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = lane_pixel_x[left_lane_inds]
    lefty = lane_pixel_y[left_lane_inds]
    rightx = lane_pixel_x[right_lane_inds]
    righty = lane_pixel_y[right_lane_inds]

    #做二次多项式拟合
    print('lefty:{},leftx:{}'.format(lefty,leftx))
    left_fit = np.polyfit(lefty, leftx, 2)
    print('righty:{},rightx:{}'.format(righty,rightx))
    right_fit = np.polyfit(righty, rightx, 2)
    
    #绘制曲线
    if visualization:
        out_img = np.dstack((binary_bev_img,binary_bev_img,binary_bev_img)) * 255
        out_img = out_img.astype('uint8')

        #根据曲线方程算出每一个y对应的x  
        ploty = np.linspace(0, h-1, h ) 
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        #车道线像素点绘制
        out_img[lane_pixel_y[left_lane_inds], lane_pixel_x[left_lane_inds]] = [255,0,0]
        out_img[lane_pixel_y[right_lane_inds], lane_pixel_x[right_lane_inds]] = [255, 0, 0]

        #车道线拟合曲线绘制
        ploty = [int(e) for e in ploty ]
        left_fitx = [int(e) for e in left_fitx]
        right_fitx = [int(e) for e in right_fitx]
        for i in range(-5,5): #只画一条线的话　太模糊了　看不出来
            draw_left_fitx = [e+i for e in left_fitx]
            draw_right_fitx = [e+i for e in right_fitx]
            out_img[ploty,draw_left_fitx] = [0,0,255]
            out_img[ploty,draw_right_fitx] = [0,0,255]
    
    return left_fit,right_fit,leftx,rightx,out_img

def warper(img, M):

    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def post_process(prediction_binary_img,origin_img):
    cv2.imwrite('prediction_binary.png',prediction_binary_img)
    #2710x3384图上的选点位置　顺序为lt,lb,rb,rt
    x = [1502, 1421, 1981, 1877]
    y = [1917, 2050, 2065, 1935]
    X = [200, 200, 550, 550] #
    Y = [200, 850, 850, 200]
    #调整世界坐标系原点的位置使得逆透视变换后车身所在车道大约位于图像中央位置
    X_offset,Y_offset = 1200,1200
    X = [e + X_offset for e in X] #
    Y = [e + Y_offset for e in Y]

    #计算逆透视图上每个像素代表的实际物理距离
    LANE_WIDTH=3.5 #需要实际测量
    # Y_POINTS_LEN=4.5 #需要实际测量
    # x_m_per_pixel = LANE_WIDTH/(x[2]-x[1])
    # y_m_per_pixel = Y_POINTS_LEN/(y[1]-y[0])
    # print('x方向每个像素代表{:.3f}m'.format(x_m_per_pixel))
    # print('y方向每个像素代表{:.3f}m'.format(y_m_per_pixel))

    src = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]))
    dst = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]],[X[2], Y[2]], [X[3], Y[3]]]))
    M = cv2.getPerspectiveTransform(src, dst)

    binary_bev_img = warper(prediction_binary_img,M)
    # cv2.imwrite('prediction_binary_bev.png',binary_bev_img)
    try:
        left_fit,right_fit,leftx,rightx,out_img = fit_line(binary_bev_img)
        # cv2.imwrite('prediction_line_bev.png',out_img)

        ##在原图上添加距离车道线的距离
        self_car_pixel=(2285,2697) 
        car_vec = np.array([self_car_pixel[0],self_car_pixel[1],1])
        car_vec_in_bird_view = M.dot(car_vec.T)
        car_vec_in_bird_view /= car_vec_in_bird_view[2]
        car_x_bev,car_y_bev = car_vec_in_bird_view[0],car_vec_in_bird_view[1]
        left_x = left_fit[0]*car_y_bev**2 + left_fit[1]*car_y_bev + left_fit[2]
        left_pixel_distance = car_x_bev - left_x
        right_x = right_fit[0]*car_y_bev**2 + right_fit[1]*car_y_bev + right_fit[2]
        right_pixel_distance = right_x - car_x_bev
        m_bev_every_pixel = LANE_WIDTH/360 #需要实际测量
        left_distance = m_bev_every_pixel * left_pixel_distance
        right_distance = m_bev_every_pixel * right_pixel_distance
        # print('bev视角下距离左车道:{}个pixel,距离右车道:{}个pixel'.format(left_pixel_distance,right_pixel_distance))
        print('bev视角下距离左车道:{:.2f}m,距离右车道:{:.2f}m'.format(left_distance,right_distance))
        
        str_left_distance = '{:.2f}'.format(left_distance)
        str_right_distance = '{:.2f}'.format(right_distance)
        str_distance = 'l:{}m,r:{}m'.format(str_left_distance,str_right_distance)
        cv2.putText(origin_img, str_distance, self_car_pixel, cv2.FONT_HERSHEY_SIMPLEX, 3,(0, 0, 255), 2, cv2.LINE_AA)


        ##绘制了车道线曲线的透视图变换到原图视角
        M_inv = cv2.getPerspectiveTransform(dst, src)
        line_img = cv2.warpPerspective(out_img, M_inv, (out_img.shape[1], out_img.shape[0]), flags=cv2.INTER_NEAREST) 
        # cv2.imwrite('prediction_line.png',line_img)
        
        #叠加到原图
        output = cv2.addWeighted(origin_img, 1, line_img, 2, 0)
        # cv2.imwrite('fit.png',output)
        return output
    except:
        print('eeeeeeeeeeeeeeeeeeeee')
        return origin_img
 
import onnx
def export_onnx():
    # change onnx input/output types
    # import onnx
    # onnx_model = onnx.load(ONNX_FILE_PATH)
    # graph = onnx_model.graph
    # in_type = getattr(graph.input[0], "type", None)
    # getattr(in_type, "tensor_type", None).elem_type = 10  # fp16
    # out_type = getattr(graph.output[0], "type", None)
    # getattr(out_type, "tensor_type", None).elem_type = 10  # fp16
    # onnx.save(onnx_model, ONNX_FILE_PATH)


    onnx_save_path='./banqiao.onnx'
    output_names = ['output']
    
    # 加载模型 不清楚原因,只能在cpu上处理 torch版本:1.13.1+cu117
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # weigths = torch.load('./runs/train/yolov7209/weights/epoch_149.pt')
    # weigths = torch.load('./runs/train/yolov7218/weights/epoch_079.pt') #model input:640
    # weigths = torch.load('./runs/train/yolov7221/weights/epoch_059.pt') #model input:1280
    weigths = torch.load('./runs/train/banqiao5/weights/epoch_099.pt') #model input:1280
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

def main():
    export_onnx()

    # filelist=[]
    # # rootdir = '/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/Camera 5'
    # rootdir = '/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record005/Camera 5'
    # details = rootdir.split('/')
    # videoname = 'apollo_{}_{}.avi'.format(details[-4],details[-2])
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
# test_img = '/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/Camera 5/170927_063845516_Camera_5.jpg'
# test_img = '/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/Camera 5/170927_063814371_Camera_5.jpg' 
# result_img = predict(test_img)
# print('result_img:{}'.format(result_img.shape))
# cv2.imwrite('./170927_063814371_Camera_5_prediction.png',result_img)

def test_pipeline():
    # test_img = '/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/Camera 5/170927_063845516_Camera_5.jpg' 
    # test_img = '/home/autocore/work_sc/datasets/lane_marking_exa# cv2.imwrite('fit.png',output)mples/road02/ColorImage/Record001/Camera 6/170927_063811892_Camera_6.jpg'
    test_img = '/home/autocore/work_sc/datasets/lane_marking_examples/road02/ColorImage/Record001/Camera 5/170927_063817253_Camera_5.jpg'    
    result_on_origin_img,result_binary_img = predict(test_img)
    origin_img = cv2.imread(test_img)
    output = post_process(result_binary_img,origin_img)
    cv2.imwrite('prediction_fit.png',output)

# test_pipeline()







