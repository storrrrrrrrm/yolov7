# python train.py --workers 8 --device 0 --batch-size 1 --data data/coco.yaml --img 640 640 --cfg cfg/multihead.yaml --weights '' --name yolov7 --hyp data/hyp.lane.yaml --notest --resume
python train.py --data data/banqiao_cam8M.yaml --img 1280 --batch-size 8 --rect --cfg cfg/multihead.yaml --hyp data/hyp.lane_banqiao.yaml --name banqiao_cam8M_1280x736  


python train.py --data data/banqiao_lane_seg.yaml --img 1280 --batch-size 16 --rect --cfg cfg/multihead_multicls.yaml --hyp data/hyp.lane_banqiao.yaml --name multihead_multicls  
