
import os
import os.path
 
 
i = 1
a = range(1,2000)
name_list = list(a)
# rootdir='/home/autocore/work_sc/datasets/apollo_lane_mark_example'


def get_file(dirname,filelist=[]):
    for dirpath, dirname, filenames in os.walk(dirname):
        for filename in sorted(filenames):
            if filename.endswith('jpg') or filename.endswith('png'):
                fullpath = os.path.join(dirpath, filename)
                filelist.append(fullpath)
                print(fullpath)

def process(rootdir,savetxt):
    filelist=[]
    get_file(rootdir,filelist)
    with open(savetxt,'w+') as f:
        for img_path in filelist:
            label_img_path = img_path.replace('Image','Label').replace('.png','_bin.png')
            print('label_img_path:{}'.format(label_img_path))
            if os.path.exists(label_img_path):
                f.writelines('{}{}'.format(img_path,'\n'))

# process('/home/autocore/work_sc/datasets/lane_marking_examples/','/home/autocore/work_sc/yolov7/coco/debug2017.txt')
process('/home/autocore/work_sc/datasets/banqiao/20230313/road03/Image','/home/autocore/work_sc/yolov7/banqiao/banqiao.txt')

