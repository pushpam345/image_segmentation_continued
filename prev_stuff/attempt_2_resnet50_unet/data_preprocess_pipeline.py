import numpy as np
from PIL import Image
import tensorflow as tf
from config import HEIGHT, WIDTH, BATCH_SIZE, CLASS_IDS
import os


import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

'''
Data laoder gives a tensor of dimension 
(BATCH_SIZE, Height, Width, 3),(BATCH__sizE,Height,Widht)
'''


def remap_mask(mask):
    mask = tf.expand_dims(mask, -1)                  # (H,W,1)
    eq = tf.equal(mask, CLASS_IDS)                   # (H,W,C)
    mask = tf.argmax(tf.cast(eq, tf.int32), axis=-1) # (H,W)
    return mask

def augment_pair(image,mask):
    seed=tf.random.experimental.stateless_split(tf.random.uniform([2], maxval=1000, dtype=tf.int32), num=1)[0]
    if tf.random.uniform(())>0.5:
        image=tf.image.stateless_random_flip_left_right(image,seed)
        mask=tf.image.stateless_random_flip_left_right(tf.expand_dims(mask, -1),seed)
        mask=tf.squeeze(mask,axis=-1)

    image=tf.image.random_brightness(image,max_delta=25.0)
    image=tf.image.random_contrast(image,lower=0.8,upper=1.2)
    image=tf.image.random_saturation(image,lower=0.8,upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    #image=tf.clip_by_value(image,0.0,1.0)

    return image,mask



def load_image_and_mask(image_path,mask_path,debug=False):
    image=tf.io.read_file(image_path)
    image=tf.image.decode_png(image,channels=3)
    image=tf.image.resize(image,(HEIGHT,WIDTH),method="bilinear")
    image=tf.cast(image,tf.float32)
    #image=tf.keras.applications.resnet50.preprocess_input(image) yaha preprcoess nhi krne ka, destroys the training pipeline

    mask=tf.io.read_file(mask_path)
    mask=tf.image.decode_png(mask,channels=1,dtype=tf.uint16)
    mask=tf.image.resize(mask,(HEIGHT,WIDTH),method="nearest")
    mask=tf.squeeze(mask,axis=-1)
    mask = remap_mask(mask)
    mask=tf.cast(mask,tf.int32)
    if debug:
        print ("Load image and mask function started ")
        print("image",type(image))  
        print("image shape:",image.shape) 
        print("mask type:",type(mask))  
        print("mask shape:",mask.shape) 
        print ("Load image and mask function ended")

    return image,mask

def get_sorted_file_paths(image_dir,mask_dir):
    image_files=sorted([os.path.join(image_dir,f)
                        for f in os.listdir(image_dir)
                        if f.endswith(".png")
                        ])
    mask_files=sorted([os.path.join(mask_dir,f)
                       for f in os.listdir(mask_dir)
                       if f.endswith(".png")
                       ])
    return image_files,mask_files

def build_dataset(image_dir, mask_dir,isTrainingDataset=True,performAugment=True):
    image_paths,mask_paths=get_sorted_file_paths(image_dir, mask_dir)
    dataset=tf.data.Dataset.from_tensor_slices(
        (image_paths,mask_paths)
    )
    if isTrainingDataset:
        dataset=dataset.shuffle(buffer_size=256)
    dataset=dataset.map(
        load_image_and_mask,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if isTrainingDataset and performAugment:
        dataset=dataset.map(
            augment_pair,num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset=dataset.map(
        lambda img, m: (
            tf.keras.applications.resnet50.preprocess_input(img),
            m
        ),
    num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset=dataset.batch(BATCH_SIZE)
    dataset=dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    image_path = "mt10000611.png"
    mask_path = "mt10000611_seg.png"

    seg_img = np.unique(np.array(Image.open(mask_path)))
    seg_img_tf = tf.io.read_file(mask_path)
    seg_img_tf = tf.image.decode_png(seg_img_tf, channels=1, dtype=tf.uint16)

    print("=====================================================================================")
    print("tensorflow loading unique values: ", tf.unique(tf.reshape(seg_img_tf, [-1])))
    print("numpy loading unique values: ", seg_img)
    print("=====================================================================================")

    train_dataset = build_dataset(
        image_dir="data/Offroad_Segmentation_Training_Dataset/train/Color_Images",
        mask_dir="data/Offroad_Segmentation_Training_Dataset/train/Segmentation",
        isTrainingDataset=True
    )

    val_dataset = build_dataset(
        image_dir="data/Offroad_Segmentation_Training_Dataset/val/Color_Images",
        mask_dir="data/Offroad_Segmentation_Training_Dataset/val/Segmentation",
        isTrainingDataset=False
    )

    for images, masks in train_dataset.take(1):
        print(images.shape)
        print(masks.shape)
        print(images.dtype)
        print(masks.dtype)
        print(tf.unique(tf.reshape(masks, [-1]))[0])
        print("*"*50)
        print(tf.reduce_min(images), tf.reduce_max(images))


