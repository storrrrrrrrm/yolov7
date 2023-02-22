
import os
import os.path
 
 
i = 1
a = range(1,2000)
name_list = list(a)
# rootdir='/home/autocore/work_sc/datasets/apollo_lane_mark_example'
rootdir = '/home/autocore/work_sc/datasets/lane_marking_examples'

def get_file(dirname,filelist=[]):
    for dirpath, dirname, filenames in os.walk(dirname):
        for filename in filenames:
            if filename.endswith('jpg'):
                fullpath = os.path.join(dirpath, filename)
                filelist.append(fullpath)
                print(fullpath)

filelist=[]
get_file(rootdir,filelist)

with open('/home/autocore/work_sc/yolov7/coco/train2017.txt','w+') as f:
    for file in filelist:
        f.writelines('{}{}'.format(file,'\n'))

# for dirpath, dirname, filename in lst_files:
#     print(dirpath)
#     # for file in filename:
#     #     name = str(name_list[i]) +'.jpg'
#     #     i += 1
#     #     print(name)
#     #     # 选中的文件的目录加文件名
#     #     src = os.path.join(dirpath, file)
#     #     # 修改之后的目录加文件名
#     #     dst = os.path.join(dirpath, name)
