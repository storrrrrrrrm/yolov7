import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.multitask_dataset import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots_banqiao import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel


def metric_lane(lane_pre,lane_targets):
    lane_pre[lane_pre > 0.7] = 1
    lane_pre[lane_pre <= 0.7] = 0
    lane_pre_cpu = lane_pre.float().cpu().detach().numpy().astype(np.float32)
    lane_targets_cpu = lane_targets.float().cpu().detach().numpy().astype(np.float32)
    intersection = np.logical_and(lane_targets_cpu, lane_pre_cpu)
    dice = (2. * intersection.sum() + 1e-10) / (lane_targets_cpu.sum() + lane_pre_cpu.sum() + 1e-10)
    # print('dice:{},type(dice)'.format(dice,type(dice)))
    return dice


def metric_lane_every_cls(lane_pre,lane_targets):
    b,c,h,w = lane_pre.shape
    ret = []
    for i in range(c):
        cls_pre = lane_pre[:,i,...] 
        cls_gt = lane_targets[:,i,...]

        gt_cls_num = (cls_gt == 1).sum()
        gt_uncls_num = (cls_gt == 0).sum()
        # print('cls_id:{},gt_cls_num:{},gt_uncls_num:{}'.format(i,gt_cls_num,gt_uncls_num))

        TP = ((cls_pre > 0.7) & (cls_gt == 1)).sum()
        FP = ((cls_pre > 0.7) & (cls_gt == 0)).sum()
        TN = ((cls_pre < 0.7) & (cls_gt == 0)).sum()
        FN = ((cls_pre < 0.7) & (cls_gt == 1)).sum()

        precision,recall = 0., 0.
        if (TP+FP)!=0:
            precision = TP / (TP + FP)  #预测为车道线点,其中多少是对的
        if (TP+FN)!=0:
            recall = TP / (TP + FN)     #真正的车道线点,其中多少被预测出来了

        # print('TP:{},FP:{},TN:{},FN:{}'.format(TP,FP,TN,FN))
        # print('precision:{},recall:{}'.format(precision,recall))

        dice = metric_lane(cls_pre,cls_gt)
        # print('dice:{}'.format(dice))

        ret.append([precision,recall,dice])
    
    return ret


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_test_dataset=False):
    # print('begin test---------------------------')

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        if trace:
            model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    # print('begin load data---------------------')
    if not training:
        # if device.type != 'cpu':
        #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images

        test_path = data['test']
        print('create dataloader from test_path:{}'.format(test_path))
        dataloader, _ = create_dataloader(test_path, imgsz, batch_size, gs, opt,
                                          prefix=colorstr('test: '),isLane=True)
    # print('end load data---------------------')
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    
    s = ('%12s') % ('Dice@0.7')
    all_pre_probs,all_lane_targets = [],[]
    for batch_i, (img, multi_targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        targets,lane_targets = multi_targets
        targets = targets.to(device)
        lane_targets = lane_targets.to(device)  #

        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            # print('inference begin')
            x,lane_seg_head_out = model(img, augment=augment)  # inference and training outputs
            # print('inference end')

            t0 += time_synchronized() - t

        # Plot images
        if plots and batch_i < 20:
            if not is_test_dataset:
                f = '{}/valid_batch{}_labels'.format(save_dir,batch_i)
                Thread(target=plot_images, args=(img, lane_targets, paths, f, names), daemon=True).start()
                f = '{}/valid_batch{}_predictions'.format(save_dir,batch_i)
                # print('save prediction:{}'.format(f))
                pre_prob = torch.sigmoid(lane_seg_head_out)
                pre_prob_mask = (pre_prob > 0.7)
                # print('lane_targets shape:{},pre_prob_mask shape:{}'.format(lane_targets.shape,pre_prob_mask.shape))
                Thread(target=plot_images, args=(img, pre_prob_mask, paths, f, names), daemon=True).start()
            elif is_test_dataset:
                f = '{}/test_batch{}_predictions'.format(save_dir,batch_i)
                # print('save prediction:{}'.format(f))
                pre_prob = torch.sigmoid(lane_seg_head_out)
                pre_prob_mask = (pre_prob > 0.7)
                # print('lane_targets shape:{},pre_prob_mask shape:{}'.format(lane_targets.shape,pre_prob_mask.shape))
                Thread(target=plot_images, args=(img, pre_prob_mask, paths, f, names), daemon=True).start()
            else:
                print('does not support')
        pre_prob = torch.sigmoid(lane_seg_head_out)
        # print('lane_targets shape:{},pre_prob shape:{}'.format(lane_targets.shape,pre_prob.shape))
        all_pre_probs.append(pre_prob)
        all_lane_targets.append(lane_targets)

    if is_test_dataset:
        return 0

    #metric
    dice = metric_lane(torch.cat(all_pre_probs,dim=0),torch.cat(all_lane_targets,dim=0))
    # print(dice)
    # Print results
    pf = '%10.4g' # print format
    print(pf % (dice))

    cls_metric_details = metric_lane_every_cls(torch.cat(all_pre_probs,dim=0),torch.cat(all_lane_targets,dim=0))

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return dice,cls_metric_details


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/road0203_hascarintrain_recttrain_arrow_exponential_decay2/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/banqiao_lane_seg.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('test'):  # run normally
        dice = test(opt.data,
                            batch_size=1,
                            imgsz=1280,
                            conf_thres=0.001,
                            iou_thres=0.7,
                            model=None,
                            single_cls=False,
                            dataloader=None,
                            save_dir='./runs/test',
                            save_json=True,
                            plots=True,
                            is_test_dataset=True)

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
