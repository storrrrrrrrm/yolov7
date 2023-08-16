#!/usr/bin/env python3

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme
import uuid

def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]

        if label.startswith('area'):
            continue
        
        if 'drivable' in label:
            continue

        if 'alternative' in label:
            continue

        #convert  "white_dotted_1" to "white_dotted"
        label = label.split('_')[0] + '_' + label.split('_')[1]

        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        # print('points:{}'.format(points))
        mask = labelme.utils.shape_to_mask(img_shape[:2], points, shape_type)

        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    # os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    # os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClassPNG"))
    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, "SegmentationClassVisualization")
        )
    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_name_to_id:{}".format(class_name_to_id))
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        # try:
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_lbl_file = osp.join(
            args.output_dir, "SegmentationClass", base + ".npy"
        )
        out_png_file = osp.join(
            args.output_dir, "SegmentationClassPNG", base + ".png"
        )
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )

        # with open(out_img_file, "wb") as f:
        #     f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        lbl, _ = shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        labelme.utils.lblsave(out_png_file, lbl)

        # np.save(out_lbl_file, lbl)

        if not args.noviz:
            viz = imgviz.label2rgb(
                lbl,
                imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
            imgviz.io.imsave(out_viz_file, viz)
        # except:
        #     print(filename,'sth wrong')


if __name__ == "__main__":
    """
    usage:./labelme2ac.py /home/sc/work/data/banqiao/test/road03/Image /home/sc/work/data/banqiao/test/road03/Labeled --labels labels.txt
    ./labelme2ac.py /home/sc/work/data/banqiao/debug /home/sc/work/data/banqiao/debug/SCRIPTS --labels labels.txt
    ./labelme2ac.py /home/sc/work/data/banqiao/banqiao_lane_seg /home/sc/work/data/banqiao/debug/SCRIPTS --labels labels.txt
    ./labelme2ac.py /mnt/data/public_datasets//banqiao/banqiao_lane_seg/Image /mnt/data/public_datasets//banqiao/banqiao_lane_seg/Label --labels labels.txt
    """
    main()
