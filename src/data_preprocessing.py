import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom as di
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from PIL import Image
from random import choice
import pickle

train_dir = "../dataset/train_images/"
test_dir = "../dataset/test_images/"

# Loading the CSV file and cleaning the data
train_csv = pd.read_csv("../dataset/train.csv")
train_csv.head(10)
train_csv.isnull().sum()


#Exploring data
print(f"<-- Difficult negative case --> \n {train_csv['difficult_negative_case'].value_counts()}\n")

print(f"<-- Cancer --> \n {train_csv['cancer'].value_counts()}\n")

# The data is very skewed.


#Function to get numpy array of the given image_id
def imageid_to_numpy(patient_id, image_id):
    path_to_images = f"{train_dir}{patient_id}/{image_id}.dcm"
    dcm_image = di.dcmread(path_to_images)
    np_array = dcm_image.pixel_array
    return np_array

#Function to copy dcm image as png
def copy_dcm_as_png(path_to_dcm, png_save_path):
    im = Image.fromarray(imageid_to_numpy(path_to_dcm))
    im.save(png_save_path)


### Work in progress
# takes df with info of a single patient and concats the corresponding images into one
def image_concat(patient_df):
    image_array = []
    for image_id in patient_df[patient_df['laterality'] == 'L']['image_id']:
        image_array.append(imageid_to_numpy(patient_df['patient_id'][0], image_id))
    
    row1 = np.concatenate(6(image_array[0], image_array[1]), axis = 1) # concat L images

    for image_id in patient_df[patient_df['laterality'] == 'R']['image_id']:
        image_array.append(imageid_to_numpy(patient_df['patient_id'][0], image_id))

    row2 = np.concatenate((image_array[0], image_array[1]), axis = 1) # concat R images

    final_image = np.concatenate((row1, row2), axis=0) # concat L and R images

    # im = Image.fromarray(final_image)
    # im = im.convert('RGB')
    # im.save('./concat_images/{}.png'.format(patient_id), compress_level = 1)

    cv2.imwrite('./concat_images/{}.png'.format(patient_df['patient_id'][0]), final_image)

    #####
    # processing is very slow for some reason
    #####


# Creating colums for n_images -> Number of images for each patient.
#                     resolution -> Dimensions of the corresponding image.
train_csv['n_images'] = [0 for _ in range(train_csv.shape[0])]
train_csv['resolution'] = [None for _ in range(train_csv.shape[0])]
train_csv['readable'] = [None for _ in range(train_csv.shape[0])]
train_csv.head(10)


# Getting the resolution for each image and the numberr of images each patient has.

try:
    processed_train_csv = pd.read_csv("processed_train.csv")
except:
    for patient in tqdm(train_csv['patient_id'].unique()):
        for image in os.listdir(train_dir + str(patient)):
            try:
                current_image = imageid_to_numpy(patient, image[:-4])
                train_csv.loc[train_csv['image_id'] == int(image[:-4]), "readable"] = True
                train_csv.loc[train_csv['image_id'] == int(image[:-4]), 'resolution'] = f"{current_image.shape[0]} {current_image.shape[1]}"
            except NotImplementedError:
                train_csv.loc[train_csv['image_id'] == int(image[:-4]), "readable"] = False
    train_csv.to_csv("./processed_train.csv")
    processed_train_csv = train_csv
    for patient in processed_train_csv["patient_id"].unique():
        n_images = processed_train_csv[processed_train_csv["patient_id"] == patient].shape[0]
        processed_train_csv.loc[processed_train_csv["patient_id"] == patient, "n_images"] = n_images
        



plt.hist(processed_train_csv["n_images"])


fig, axes = plt.subplots(1, 1, figsize=(15, 8))
axes.hist(processed_train_csv["resolution"])




plt.hist(processed_train_csv.loc[processed_train_csv["cancer"] == 1]["resolution"])


processed_train_csv[processed_train_csv["resolution"] == "4096 3328"]


# rand_patient = choice(processed_train_csv.loc[processed_train_csv['resolution'] == "4096 3328"]["patient_id"].unique())
rand_patient = choice(processed_train_csv.loc[processed_train_csv['patient_id'] == 28844]['image_id'].unique())

rand_patient


# rand_patient = choice(processed_train_csv.loc[processed_train_csv['resolution'] == "4096 3328"]["patient_id"].unique())
# rand_patient = 46905
# rand_image = choice(processed_train_csv.loc[processed_train_csv['patient_id'] == rand_patient]['image_id'].unique())
# print(processed_train_csv.loc[processed_train_csv['image_id'] == rand_image]['cancer'])

# image = imageid_to_numpy(rand_patient, rand_image) * 10

# image1 = cv2.resize(image, (500, 500))

# cv2.imshow("output", image1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


plt.hist(processed_train_csv["age"])


cancer_train_df = processed_train_csv[processed_train_csv["cancer"] == 1]


train_df= cancer_train_df.append( processed_train_csv[processed_train_csv["cancer"] == 0].sample(n = cancer_train_df.shape[0], replace=False))



train_df = train_df.sample(frac = 1).reset_index()
train_df.head()


#drop columns
train_df_ids = train_df["image_id"]
train_df_ids.shape


train_labels = train_df["cancer"]
train_labels.shape


dimension = 100

try:
    with open("./xtrain3.pkl", "rb") as f:
        train_x = pickle.load(f)
except:
    ## Creating image numpy array of size N examples X dimensions ** 2 (2316, 10000)

    train_x = np.zeros((train_labels.shape[0],  dimension, dimension))

    for row in tqdm(train_df.iterrows()):
        image_array = imageid_to_numpy(row[1]["patient_id"], row[1]["image_id"])
        image_array = cv2.resize(image_array, (dimension, dimension)) 
        # print(np.max(image_array), end=" <> ")
        normalized_array = image_array / np.max(image_array)
        # train_x = np.append(train_x, image_array.flatten(), 1)
        # train_x = np.concatenate((train_x, normalized_array.flatten()), 1)
        train_x[row[0]] = normalized_array
    

    with open("./xtrain3.pkl", "wb") as f:
        pickle.dump(train_x, f)


train_x.shape[1:]


