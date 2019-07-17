import json
import glob
import tqdm 
import cv2
import argparse
import os

categories = ['Cyclist','Tram','Person_sitting','Truck','Pedestrian','Van','Car','Misc','DontCare']
categoriesv2 = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 
              'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket','bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

def generate_json_categories(data):
    #Loop over each category and add them
    for i,category in enumerate(categories):
        data['categories'].append({
            'supercategory' : category,
            'id' : i+1,
            'name' : category 
        })
        
def id_converter(id,index):
    return int(id)*10000+index

def parse_line(data,line,image_id,index):
    args=line.split()
    
    left=float(args[4])
    top=float(args[5])
    right=float(args[6])
    bottom=float(args[7])
    
    #convert from kitti to COCO bbox format: (left,top,right,bottom) -> (x,y,w,h)
    bbox         = [round(left),round(top),round(abs(right-left)),round(abs(bottom-top))] 
    dimensions   = [float(args[8]),float(args[9]),float(args[10])]
    locations    = [float(args[11]),float(args[12]),float(args[13])]
    ry           = float(args[14])
    area         = int(round(bbox[3]*bbox[2]))
    alpha        = float(args[3])
    cat_id       = categories.index(args[0])+1
    
    data['annotations'].append({
        'area' : area, #force int format
        'iscrowd': 0,
        'image_id': int(image_id),
        'bbox' : bbox,
        'category_id' : cat_id,
        'id' : id_converter(image_id,index),
        'segmentation' : [[]], #empty, not used for KITTI
        'alpha': alpha,
        'dimensions':dimensions,
        'location':locations,
        'ry':ry
    })
    
    return bbox

def url_generator(url):
    return 'http://totally.realwebsite.com/'+url

def get_date(path):
    return os.path.getmtime(path) # just making sure youre using the correct operating system. 

def per_image(data,image,annotation,image_location,image_id,is_test=False):
    #Gather everything below for each image
    h,w,c = image.shape
    ann_id = annotation
    category_id = 0
    category = ""
    url = url_generator(annotation) # these need to be made
    
    file = open(annotation,"r")
    if not is_test:
        for i,line in enumerate(file.readlines()):
            parse_line(data,line,image_id,i)
        
    data['images'].append({
        'license' : 1,
        'file_name' : image_location[-10:],
        'height' : h,
        'width' : w,
        'date_captured' : get_date(image_location), # these need to be made
        'id' : int(image_id)
    })
def per_imagev2(data,image,image_location,image_id,is_test=False):
    #Gather everything below for each image
    h,w,c = image.shape
    category_id = 0
    category = ""
        
    data['images'].append({
        'license' : 1,
        'file_name' : image_location[-10:],
        'height' : h,
        'width' : w,
        'date_captured' : get_date(image_location), # these need to be made
        'id' : int(image_id)
    })
    
def generate_json(imagedir,annodir,resultname,slicer):
    #Execute once
    images=glob.glob(imagedir+'/*.png')
    annotations = glob.glob(annodir+'/*.txt')
    images.sort()
    images=eval('images'+slicer)
    annotations.sort()
    annotations=eval('annotations'+slicer)
    assert len(images)==len(annotations),"got len"+str(len(images))+" and "+str(len(annotations))
    data={}#make this ordered?
    data['info'] = {'description': 'Scott','url':'uhh','version':'6.9','year':2014,} 
    data['licenses'] = []  
    data['images'] = []  
    data['annotations'] = []  
    data['categories'] = []
    
    data['licenses'].append({  
        'url': 'Scott',
        'id': 1,
        'name': 'Nebraska'
    })

    generate_json_categories(data)
    
    for i,name in enumerate(tqdm.tqdm(images)): #loop through images, display loading bar
        per_image(data, cv2.imread(str(name)),str(annotations[i]),name,name[-10:-4])

    with open('data/kitti/annotations/{}.json'.format(resultname), 'w') as fp:
        json.dump(data, fp)
        
def generate_jsonv2(imagedir,resultname,slicer): 
    #Execute once
    images=glob.glob(imagedir+'/*.png')
    images.sort()
    images=eval('images'+slicer)
    data={}#make this ordered?
    data['info'] = {'description': 'Scott','url':'uhh','version':'6.9','year':2014,} 
    data['licenses'] = []  
    data['images'] = []   
    data['categories'] = []
    
    data['licenses'].append({  
        'url': 'Scott',
        'id': 1,
        'name': 'Nebraska'
    })
    
    generate_json_categories(data)
        
    for i,name in enumerate(tqdm.tqdm(images)): #loop through images, display loading bar
        per_imagev2(data,cv2.imread(str(name)),name,name[-10:-4])

    with open('data/kitti/annotations/{}.json'.format(resultname), 'w') as fp:
        json.dump(data, fp)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Kitti into Coco Format')
    parser.add_argument('--mode',help='use on training or inference',default='training')
    parser.add_argument('--imagedir',help='path to directory of images',default='/home/ubuntu/kitti-3d-detection-unzipped/training/image_2')
    parser.add_argument('--annodir',help='path to directory of annotations',default='/home/ubuntu/kitti-3d-detection-unzipped/training/label_2')
    parser.add_argument('--name',help='just give a name for putting the data in',default='all_{}_data')
    parser.add_argument('--slice',help='how to slice the dataset (eg giving 0 300 will give the first 300)',default=[],nargs='+',type=int,metavar='')
    parser.add_argument('--dataset',help='use on "kitti" or "coco"',default='kitti')
    args=parser.parse_args()
    if args.dataset.lower()=='coco':
        categories=categoriesv2
    assert len(args.slice)<=3,'too many values'
    assert len(args.slice) is not 1,'invalid indicies'
    if args.slice:
        slicer = '['
        for i in args.slice:
            slicer += str(i)+':'
        if(slicer[-1]==':'):
            slicer=slicer[:-1]
        slicer += ']'
    else:
        slicer=''
    name = args.name.format(args.mode)
    if args.mode == 'training':
        generate_json(args.imagedir,args.annodir,name,slicer)
    else:
        generate_jsonv2(args.imagedir,name,slicer)