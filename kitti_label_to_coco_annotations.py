# this script converts kitti images and labels to coco format
import json
import glob
import tqdm 
import cv2
import argparse
import os
import math

KITTI_CATEGORIES = ['Cyclist','Tram','Person_sitting','Truck','Pedestrian','Van','Car','Misc','DontCare'] # categories for kitti
COCO_CATEGORIES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog','horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket','bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'] # categories from coco
    
def generate_json(image_dir, label_dir, output_filename, slicer, categories, is_train=True, remove_DontCare=1):
    output_dict = {}
    output_dict['images'] = []  
    output_dict['categories'] = []
    
    images = glob.glob(image_dir+ '/*.png')#PLEASE DONT ADD OS.PATH.JOIN. PLEASE.
    images.sort()
    images = eval('images'+slicer) # use slicer to filter which images to generate
    
    if is_train:
        
        output_dict['annotations'] = []  
        annotations_txts = glob.glob(label_dir+'/*.txt')
        annotations_txts.sort()
        annotations_txts = eval('annotations_txts'+slicer) # use slicer to filter which annotations to generate
        assert len(images)==len(annotations_txts), "got images and annotations len"+str(len(images))+" and "+str(len(annotations_txts))
        
    for idx, category in enumerate(categories):
        if remove_DontCare and category == 'DontCare':
            continue
        output_dict['categories'].append({
            'id' : idx,
            'name' : category
        })
        
    for i, image_path in enumerate(tqdm.tqdm(images,desc='going over images')): # loop through images, display loading bar
        
        h, w, c = cv2.imread(image_path).shape
        image_id = image_path[-10:-4]
        
        output_dict['images'].append({
            'file_name' : image_path[-10:],
            'height' : h,
            'width' : w,
            'id' : int(image_id)
        })
        
        if is_train:

            with open(annotations_txts[i], "r") as txt_file:

                for j, line in enumerate(txt_file.readlines()):

                    args = line.split()
                    if remove_DontCare and args[0]=='DontCare':
                        continue
                    left = math.floor(float(args[4]))
                    top = math.floor(float(args[5]))
                    right = math.ceil(float(args[6]))
                    bottom = math.ceil(float(args[7]))

                    assert right > left, "right is not bigger than left:"+image_path
                    assert bottom > top, "bottom is not bigger than top:"+image_path

                    # convert from kitti to COCO bbox format: (left,top,right,bottom) -> (x,y,w,h)
                    bbox         = [left, top, right-left, bottom-top]

                    dimensions   = [float(args[8]),float(args[9]),float(args[10])]
                    locations    = [float(args[11]),float(args[12]),float(args[13])]
                    ry           = float(args[14])
                    area         = bbox[2]*bbox[3]
                    alpha        = float(args[3])
                    cat_id       = categories.index(args[0])
                    trucation    = int(float(args[1]))
                    occusion     = int(float(args[2]))
                    
                    
                    output_dict['annotations'].append({
                        'area': area,
                        'image_id': int(image_id),
                        'bbox' : bbox,
                        'category_id': cat_id,
                        'id' : int(image_id)*1000+j,
                        'alpha': alpha,
                        'dimensions':dimensions,
                        'location':locations,
                        'ry':ry, 
                        'truncated': trucation,
                        'occluded': occusion,
                    })

    with open('{}.json'.format(output_filename), 'w') as fp:
        json.dump(output_dict, fp)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Kitti into Coco Format')
    parser.add_argument('--mode', help='either training(default) or testing', default='training') # subdir of the dataset
    parser.add_argument('--image_dir', help='path to directory of images', default='/home/ubuntu/kitti-3d-detection-unzipped/training/image_2',metavar='path')
    parser.add_argument('--label_dir', help='path to directory of annotations', default='/home/ubuntu/kitti-3d-detection-unzipped/training/label_2',metavar='path')
    parser.add_argument('--slice', help='give start and end (exclusive) index, how to slice the dataset (eg giving 0 300 will give the first 300)', default=[], nargs='+' , type=int,metavar=('start','end'))
    parser.add_argument('--category', help='use on "kitti" or "coco"', default='kitti',metavar='category')
    parser.add_argument('--output_filename', help='output filename, the output will filename will be in <output_filename>.json, defaults to <category>_<mode>_<slice>.json', default=None,metavar='Name')
    parser.add_argument('--output_filepath',help='this is where you want to put your json generated, eg <output_filepath><output_filename>, defaults to current dir',default='',metavar='path')
    parser.add_argument('--remove_DontCare',help='remove kitti labels with category DontCare, 1: yes(default), 0:no',default=1, type=int)
    args=parser.parse_args()
    
    assert len(args.slice) < 4, 'too many values to slice with'
    
    if len(args.slice) > 0:
        slicer = '['
        for i in args.slice:
            slicer += str(i)+':'
        if(slicer[-1]==':'): # remove the last colon
            slicer=slicer[:-1]
        slicer += ']'
    else:
        slicer=''
        
    generate_json(args.image_dir, args.label_dir, \
                  args.output_filepath+(args.output_filename if (args.output_filename is not None) else (args.category+'_'+args.mode+'_'+(slicer if args.slice else 'all'))),\
                  slicer,\
                  COCO_CATEGORIES if args.category.lower()=='coco' else KITTI_CATEGORIES,\
                  args.mode.lower() == 'training',
                  args.remove_DontCare == 1)