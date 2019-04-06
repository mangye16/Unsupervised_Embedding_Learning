import numpy as np
import scipy.io as sio
import os
from PIL import Image, ImageChops
from tqdm import tqdm

#download from 
image_url = "http://imagenet.stanford.edu/internal/car196/car_ims.tgz"
annotation_url = "http://imagenet.stanford.edu/internal/car196/cars_annos.mat"

#cut white margin
def trim(im):
    bg = Image.new(im.mode, im.size, 'white')
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -55)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

data_dir = '/home/datasets/prml/computervision/classification/car196/'

training_img_list = []
validation_img_list = []

training_label_list = []
validation_label_list = []
fix_image_width = 256
fix_image_height = 256

annotation = sio.loadmat(data_dir+'cars_annos.mat')
annotation = annotation['annotations'][0]
for label in tqdm(annotation):
    image_name, left, top, right, bottom, class_id, test_flag = label
    image_name = image_name[0]
    class_id = class_id[0][0]
    #print(image_name,class_id)
    img = Image.open(data_dir+image_name)
    #img = trim(img)
    img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
    pix_array = np.array(img)
    if len(pix_array.shape) == 2:
        pix_array.resize((pix_array.shape[0], pix_array.shape[1], 1))
        pix_array = np.repeat(pix_array, 3, 2)
    if pix_array.shape[2]>3:
        pix_array = pix_array[:,:,:3]
    if class_id <=98:
        training_img_list.append(pix_array)
        training_label_list.append(class_id)
    else:
        validation_img_list.append(pix_array)
        validation_label_list.append(class_id)

training_img = np.array(training_img_list)
training_label = np.array(training_label_list)
print(training_img.shape)
print(training_label.shape)
np.save(data_dir + 'training_car196_256resized_img.npy', training_img)
np.save(data_dir + 'training_car196_256resized_label.npy', training_label)
validation_img = np.array(validation_img_list)
validation_label = np.array(validation_label_list)
print(validation_img.shape)
print(validation_label.shape)
np.save(data_dir + 'validation_car196_256resized_img.npy', validation_img)
np.save(data_dir + 'validation_car196_256resized_label.npy', validation_label)
