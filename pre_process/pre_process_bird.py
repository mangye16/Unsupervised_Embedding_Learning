import numpy as np
import os
from PIL import Image, ImageChops
from tqdm import tqdm

#download from 
image_url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"

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
    
data_dir = '/home/datasets/prml/computervision/classification/cub200/'

training_img_list = []
validation_img_list = []

training_label_list = []
validation_label_list = []
fix_image_width = 256
fix_image_height = 256
with open(data_dir+'CUB_200_2011/image_class_labels.txt', 'r') as label_file:
    with open(data_dir+'CUB_200_2011/images.txt', 'r') as image_file:
        for label, image in tqdm(zip(label_file,image_file)):
            idx, class_id = label.split(' ')
            idx, file_name = image.split(' ')
            class_id = int(class_id[:-1])
            file_name = file_name[:-1]
            #print(class_id, file_name)
            img = Image.open(data_dir+'CUB_200_2011/images/'+file_name)
            img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
            #img = trim(img)
            #img = pad(img)
            pix_array = np.array(img)
            if len(pix_array.shape) == 2:
                pix_array.resize((pix_array.shape[0], pix_array.shape[1], 1))
                pix_array = np.repeat(pix_array, 3, 2)
            if class_id <=100 :
                training_img_list.append(pix_array)
                training_label_list.append(class_id)
            else:
                validation_img_list.append(pix_array)
                validation_label_list.append(class_id)

training_img = np.array(training_img_list)
training_label = np.array(training_label_list)
print(training_img.shape)
print(training_label.shape)
np.save(data_dir + 'training_cub200_256resized_img.npy', training_img)
np.save(data_dir + 'training_cub200_256resized_label.npy', training_label)
validation_img = np.array(validation_img_list)
validation_label = np.array(validation_label_list)
print(validation_img.shape)
print(validation_label.shape)
np.save(data_dir + 'validation_cub200_256resized_img.npy', validation_img)
np.save(data_dir + 'validation_cub200_256resized_label.npy', validation_label)
