import json
import glob
import os
import tqdm 
import tokenize
import uuid
import cv2
import copy

categories = ['Cyclist','Tram','Person_sitting','Truck','Pedestrian','Van','Car','Misc','DontCare']

def generate_json_categories(data):
    #Loop over each category and add them
    for i,category in enumerate(categories):
        data['categories'].append({
            'supercategory' : category,
            'id' : i,
            'name' : category 
        })
        
def id_generator(id):
        
    return id

def parse_line(data,line,image_id,index):#this is meant to parse a single line of the text file annotation, please complete
    args=line.split()
    ''' figure out how to take bbox from the tokenized string'''
    left=float(args[4])
    top=float(args[5])
    right=float(args[6])
    bottom=float(args[7])
    
    bbox = []
    bbox.append(round(left))
    bbox.append(round(top))
    bbox.append(round(abs(right-left)))
    bbox.append(round(abs(bottom-top)))
    dimensions=[]
    dimensions.append(float(args[8]))
    dimensions.append(float(args[9]))
    dimensions.append(float(args[10]))
    locations=[]
    locations.append(float(args[11]))
    locations.append(float(args[12]))
    locations.append(float(args[13]))
    data['annotations'].append({
        'area' : int(round(bbox[3]*bbox[2])),#force int format
        'iscrowd': 0,
        'image_id': int(image_id),
        'bbox' : bbox,
        'category_id' : categories.index(args[0])+1,
        'id' : int(image_id)*10000+index,
        'segmentation' :[[float(args[8]),float(args[9])]],
        'alpha':float(args[3]),
        'dimensions':dimensions,
        'location':locations,
        'ry':float(args[14])
    })
    
    return bbox

def url_generator(url):
    return 'http://totally.realwebsite.com/'+url

def get_date():
    return '4/20/2019'
def per_image(data,image,annotation,image_location,image_id,is_test=False):
    #Loop over everything below for each image
    h,w,c = image.shape
    date = get_date()
    ann_id = annotation
    image_id = image_id
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
        'date_captured' : get_date(), # these need to be made
        'id' : int(image_id)
        #'url': url,
    })

def generate_json(imagedir,annodir):
    #Execute once
    images=glob.glob(imagedir+'/*.png')
    annotations = glob.glob(annodir+'/*.txt')
    images.sort()
    annotations.sort()
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
    
    for i,category in enumerate(categories):
        data['categories'].append({
            'supercategory': category,
            'id': i+1,
            'category': category
        })
    for i,name in enumerate(tqdm.tqdm(images)):
        per_image(data,cv2.imread(str(name)),str(annotations[i]),name,name[-10:-4])

    with open('data/coco/annotations/everything.json', 'w') as fp:
        json.dump(data, fp)
    
        

    


#current function
generate_json("../../kitti-3d-detection-unzipped/training/image_2","../../kitti-3d-detection-unzipped/training/label_2")