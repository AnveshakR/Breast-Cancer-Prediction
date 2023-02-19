
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom as di
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from PIL import Image
import threading


train_dir = "../dataset/train_images/"


#Function to get numpy array of the given image_id
def imageid_to_numpy(patient_id, image_id):
    path_to_images = train_dir + str(patient_id) + "/" + str(image_id) + ".dcm"
    dcm_image = di.dcmread(path_to_images)
    np_array = dcm_image.pixel_array
    return np_array

#Function to save dcm image as png
def save_dcm_as_png(path_to_dcm, png_save_path):
    im = Image.fromarray(imageid_to_numpy(path_to_dcm)*128)
    im.save(png_save_path)

# Loading the CSV file and cleaning the data
train_csv = pd.read_csv("../dataset/train.csv")






def label_data(indices):
    print(indices)
    for patient in tqdm(train_csv['patient_id'].unique()[indices[0]:indices[1]]):
        n_images = 0
        for image in os.listdir(train_dir + str(patient)):
            n_images += 1
            try:
                current_image = imageid_to_numpy(patient, image[:-4])
                train_csv.loc[train_csv['image_id'] == int(image[:-4]), "readable"] = True
                train_csv.loc[train_csv['image_id'] == int(image[:-4]), 'resolution'] = f"{current_image.shape[0]} {current_image.shape[1]}"
            except NotImplementedError:
                train_csv.loc[train_csv['image_id'] == int(image[:-4]), "readable"] = False
        
    train_csv.loc[train_csv['patient_id'] == int(patient), 'n_images'] = n_images

# n_threads = 16

# thread_array = [None] * n_threads

# len_data = len(train_csv['patient_id'].unique()) -1

# thread_workload = len_data / n_threads

# for i in range(n_threads):
#     thread_array[i] = threading.Thread(target=label_data, args=(i * thread_workload, (i+1) * thread_workload ))

# label_data(len_data - 1 - len_data % n_threads, len_data)





import concurrent.futures

n_threads = 16

thread_array = [None] * n_threads

len_data = len(train_csv['patient_id'].unique()) -1

thread_workload = len_data / n_threads


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(label_data, [(i *thread_workload, (i+1) * thread_workload) for i in range(n_threads)])


train_csv.to_csv("./processed_train.csv")






