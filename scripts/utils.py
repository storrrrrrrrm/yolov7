
import os
import os.path

def imgpath2labelpath(img_path):
    label_path = img_path.replace('Image','Label/SegmentationClassPNG')
    return label_path

 
i = 1
a = range(1,2000)
name_list = list(a)
# rootdir='/home/autocore/work_sc/datasets/apollo_lane_mark_example'

def sort_by_time(file_name):
    """
    文件名是数字｀
    """
    file_name_num = file_name[:-4]
    nums = file_name_num.split('_')
   
    return (int(nums[0]),int(nums[1]))

def get_file(dirname,filelist=[],sortkey=None):
    for dirpath, dirname, filenames in os.walk(dirname):
        for filename in sorted(filenames,key=sortkey):
            if filename.endswith('jpg') or filename.endswith('png'):
                fullpath = os.path.join(dirpath, filename)
                filelist.append(fullpath)
                print(fullpath)

def gen_traintxt(rootdir,savetxt):
    filelist=[]
    get_file(rootdir,filelist)
    with open(savetxt,'w+') as f:
        for img_path in filelist:
            label_img_path = imgpath2labelpath(img_path)
            print('img_path:{}'.format(img_path))
            print('label_img_path:{}'.format(label_img_path))
            if os.path.exists(label_img_path):
                f.writelines('{}{}'.format(img_path,'\n'))

def gen_testtxt(rootdir,savetxt):
    filelist=[]
    get_file(rootdir,filelist,sort_by_time)
    with open(savetxt,'w+') as f:
        for img_path in filelist:
            f.writelines('{}{}'.format(img_path,'\n'))


def remove_no_label(rootdir):
    for dirpath, dirname, filenames in os.walk(rootdir):
        for filename in sorted(filenames):
            if filename.endswith('jpg') or filename.endswith('png'):
                fullpath = os.path.join(dirpath, filename)
                suffix = filename[-3:]
                label_name = fullpath.replace(suffix,'json')
                if os.path.exists(label_name):
                    print(label_name)
                else:
                    print('remove {}'.format(fullpath))
                    os.remove(fullpath)

def copy_label(srcdir,dstdir):
    import shutil
    for dirpath, dirname, filenames in os.walk(srcdir):
        for filename in sorted(filenames):
            if filename.endswith('jpg') or filename.endswith('png'):
                fullpath = os.path.join(dirpath, filename)
                suffix = filename[-3:]
                label_name = fullpath.replace(suffix,'json')
                if os.path.exists(label_name):
                    print(label_name)
                    shutil.copy(fullpath, dstdir)
                    shutil.copy(label_name, dstdir)                    
                else:
                    # print('remove {}'.format(fullpath))
                    pass

#把原先标注的road03的标注图的颜色改掉
def change_colcor_road03():
    COLOR_MAP = {
    'white_solid' : (0,0,128),
    'white_dotted' : (0,128,0),
    'forward_arrow' : (128,0,128),
    'diversion_line' : (0,128,64)
    }

    solid = (43,173,180)
    dotted = (8,35,142)

    srcdir = '/mnt/data/public_datasets/banqiao/20230418/road03/Label'
    import cv2
    import numpy as np
    for dirpath, dirname, filenames in os.walk(srcdir):
        for filename in sorted(filenames):
            if filename.endswith('jpg') or filename.endswith('png'):
                fullpath = os.path.join(dirpath, filename)

                img = cv2.imread(fullpath)
                h_idx,w_idx = np.where( (img == solid).all(axis=2) )
                img[h_idx,w_idx,:] = COLOR_MAP['white_solid']

                h_idx,w_idx = np.where( (img == dotted).all(axis=2) )
                img[h_idx,w_idx,:] = COLOR_MAP['white_dotted']

                cv2.imwrite(fullpath,img)


#把原先标注的road03的直路图片拷贝过来
def copy_road03():
    srcdir = '/mnt/data/public_datasets/banqiao/20230418/road03/Image'
    dstdir = '/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Image'
    import shutil
    for dirpath, dirname, filenames in os.walk(srcdir):
        for filename in sorted(filenames):
            if filename.endswith('jpg') or filename.endswith('png'):
                fullpath = os.path.join(dirpath, filename)
                suffix = filename[-3:]
                label_name = fullpath.replace(suffix,'json')
                if os.path.exists(label_name):
                    print(label_name)
                    shutil.copy(fullpath, dstdir)

                    label_img_fullpath = fullpath.replace('Image','Label').replace('.png','_bin.png')
                    

                    new_label_img_fullpath = dstdir.replace('Image','Label/SegmentationClassPNG/') + filename
                    print(new_label_img_fullpath)

                    shutil.copy(label_img_fullpath, new_label_img_fullpath)                    
                else:
                    # print('remove {}'.format(fullpath))
                    pass




if __name__ == '__main__':
    # copy_label('/mnt/data/public_datasets/banqiao/20230418/road02/Image','/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Image')
    # copy_label('/mnt/data/public_datasets/banqiao/20230418/to_be_labeled/0','/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Image')
    # copy_label('/mnt/data/public_datasets/banqiao/20230418/to_be_labeled/1','/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Image')
    # copy_label('/mnt/data/public_datasets/banqiao/20230418/to_be_labeled/2','/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Image')
    # copy_label('/mnt/data/public_datasets/banqiao/20230418/to_be_labeled/3','/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Image')
    # copy_label('/mnt/data/public_datasets/banqiao/20230418/to_be_labeled/4','/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Image')
    # copy_label('/mnt/data/public_datasets/banqiao/20230418/to_be_labeled/5','/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Image')
    # copy_label('/mnt/data/public_datasets/banqiao/20230418/to_be_labeled/6','/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Image')

    gen_traintxt('/mnt/data/public_datasets/banqiao/banqiao_lane_seg/Image', \
                 '/mnt/data/sc/yolov7/banqiao/banqiao_lane_seg.txt')

    # gen_testtxt('/mnt/data/public_datasets/banqiao/test/cam0', \
    #              '/mnt/data/sc/yolov7/banqiao/banqiao_lane_seg_test.txt')

    # change_colcor_road03()
    # copy_road03()

# process('/home/autocore/work_sc/datasets/lane_marking_examples/','/home/autocore/work_sc/yolov7/coco/debug2017.txt')
# process('/home/autocore/work_sc/datasets/banqiao/20230313/road03/Image','/home/autocore/work_sc/yolov7/banqiao/banqiao.txt')
# process('/home/autocore/work_sc/datasets/banqiao/20230313/','/home/autocore/work_sc/yolov7/banqiao/banqiao.txt')
# process('/home/autocore/work_sc/datasets/banqiao/20230418/','/home/autocore/work_sc/yolov7/banqiao/banqiao_cam8M.txt')
# process('/home/autocore/work_sc/datasets/banqiao/20230418/','/home/autocore/work_sc/yolov7/banqiao/banqiao_cam8M.txt')

