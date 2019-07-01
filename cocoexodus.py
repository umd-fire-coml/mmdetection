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

def url_generator(url):
    return 'http://totally.realwebsite.com/'+url

def get_date():
    return '4/20/2019'
def per_image(data,image,image_location,image_id,is_test=False):
    #Loop over everything below for each image
    h,w,c = image.shape
    date = get_date()
    image_id = image_id
    category_id = 0
    category = ""
        
    data['images'].append({
        'license' : 1,
        'file_name' : image_location[-10:],
        'height' : h,
        'width' : w,
        'date_captured' : get_date(), # these need to be made
        'id' : int(image_id)
        #'url': url,
    })

def generate_json(imagedir):
    #Execute once
    images=glob.glob(imagedir+'/*.png')
    images.sort()
    images=images[0:100]
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
    
    for i,category in enumerate(categories):
        data['categories'].append({
            'supercategory': category,
            'id': i+1,
            'category': category
        })
    for i,name in enumerate(tqdm.tqdm(images)):
        per_image(data,cv2.imread(str(name)),name,name[-10:-4])

    with open('data/coco/annotations/makeonethatmakessence.json', 'w') as fp:
        json.dump(data, fp)

#current function
generate_json("../../kitti-3d-detection-unzipped/testing/image_2")