import numpy as np
import scipy.io as sio
import os
from PIL import Image, ImageChops
from tqdm import tqdm


#download from 
image_url = "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"

#cut white margin
def trim(im):
    bg = Image.new(im.mode, im.size, 'white')
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -55)
    bbox = diff.getbbox()
    if bbox:
        a = max(0,bbox[0]-20)
        b = max(0,bbox[1]-20)
        c = min(im.size[0],bbox[2]+20)
        d = min(im.size[1],bbox[3]+20)
        bbox = (a,b,c,d)
        return im.crop(bbox)
    return im

def pad(im):
    if im.size[0]>im.size[1]:
        im = im.resize((fix_image_width, fix_image_height*im.size[1]/im.size[0]), Image.ANTIALIAS)
    elif im.size[1]>im.size[0]:
        im = im.resize((fix_image_width*im.size[0]/im.size[1], fix_image_height), Image.ANTIALIAS)
    else:
        im = im.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
    
    new_im = Image.new(im.mode,(fix_image_width, fix_image_height), 'white')

    new_im.paste(im, ((fix_image_width-im.size[0])/2,
                      (fix_image_height-im.size[1])/2))
    return new_im

data_dir = '/home/datasets/prml/computervision/classification/ebay/'

training_img_list = []
validation_img_list = []

training_label_list = []
validation_label_list = []
fix_image_width = 256
fix_image_height = 256
index = 0
with open(data_dir+'Ebay_train.txt', 'r') as label_file:
   for info in tqdm(label_file):
       if index == 0:
           index = index + 1
           continue
       img_idx, class_id, _, file_name = info.split(' ')
       class_id = int(class_id)
       file_name = file_name[:-1]
       #print(class_id, file_name)
       img = Image.open(data_dir+file_name)
       img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
       #img = trim(img)
       #img = pad(img)
       pix_array = np.array(img)
       if len(pix_array.shape) == 2:
           pix_array.resize((pix_array.shape[0], pix_array.shape[1], 1))
           pix_array = np.repeat(pix_array, 3, 2)
       training_img_list.append(pix_array)
       training_label_list.append(class_id)

training_img = np.array(training_img_list)
training_label = np.array(training_label_list)
print(training_img.shape)
print(training_label.shape)
np.save(data_dir + 'training_ebay_256resized_img.npy', training_img)
np.save(data_dir + 'training_ebay_256resized_label.npy', training_label)
training_img = None
training_label = None

index = 0
with open(data_dir+'Ebay_test.txt', 'r') as label_file:
    for info in tqdm(label_file):
        if index == 0:
            index = index + 1
            continue
        img_idx, class_id, _, file_name = info.split(' ')
        class_id = int(class_id)
        file_name = file_name[:-1]
#       print(class_id, file_name)
        img = Image.open(data_dir+file_name)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        #img = trim(img)
        #img = pad(img)
        pix_array = np.array(img)
        if len(pix_array.shape) == 2:
            pix_array.resize((pix_array.shape[0], pix_array.shape[1], 1))
            pix_array = np.repeat(pix_array, 3, 2)
        validation_img_list.append(pix_array)
        validation_label_list.append(class_id)

validation_img = np.array(validation_img_list)
validation_label = np.array(validation_label_list)
print(validation_img.shape)
print(validation_label.shape)
np.save(data_dir + 'validation_ebay_256resized_img.npy', validation_img)
np.save(data_dir + 'validation_ebay_256resized_label.npy', validation_label)
