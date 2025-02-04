import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from featup.util import norm, unnorm
from featup.plotting import plot_feats, plot_lang_heatmaps
from dataloader import SpinalCordDataset, SpinalCordTransform
# from model import SegmentationCNN
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from functools import partial
from trainer_unet import run_training
from argparse import Namespace
import sys
from tqdm import tqdm
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_norm = True
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=use_norm).to(device)
# model_2 = SegmentationCNN().to(device)

type_res = "hr"
folder = "/media/sarvagya-pc/2TB HDD/Balgrist/GM_mask/new_data/"
# folder_gm = "/media/sarvagya-pc/2TB HDD/Balgrist/GM_mask/01_data/"
folder_test = "/home/sarvagya-pc/Desktop/Balgrist_neuroimg/for_test/"
save_folder_train = "/home/sarvagya-pc/Desktop/Balgrist_neuroimg/featup/data/GM/"+type_res+"/train"
save_folder_val = "/home/sarvagya-pc/Desktop/Balgrist_neuroimg/featup/data/GM/"+type_res+"/val"
save_folder_test = "/home/sarvagya-pc/Desktop/Balgrist_neuroimg/featup/data/GM/"+type_res+"/test"
subs = sorted(os.listdir(folder))
print(subs)



image_nifti = []
label_nifti = []
for sub in subs:
    sessions = sorted([ses for ses in os.listdir(folder+sub+'/') if "ses-" in ses])
    for ses in sessions:
        file_image = glob(folder+sub+'/'+ses+'/anat/*_acq-lumbarMEGRE3D_desc-crop_T2starw.nii')
        file_label = glob(folder+sub+'/'+ses+'/anat/*_acq-lumbarMEGRE3D_seg-manual_label-GM_mask_cr.nii')
        
        if len(file_label)==0:
            print("this subject is empty "+sub)
            pass
        else:
            print(file_image[0])
            # print(file)
            print(file_image[0].split('/')[-1].split("_")[0])
            if file_image[0].split('/')[-1].split("_")[0] != sub:
                print("error with "+sub)
            else:
                image_nifti.append(file_image[0])
                label_nifti.append(file_label[0])

X_train, X_test, y_train, y_test = train_test_split(image_nifti, label_nifti, test_size=0.2, random_state=1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


image_paths_train = X_train
label_paths_train = y_train

image_paths_val = X_test
label_paths_val = y_test

# # Create Dataset and DataLoader
# transform = SpinalCordTransform(flip_prob=0.5)
# dataset = SpinalCordDataset(image_paths=image_paths, label_paths=label_paths, transform=transform)

# data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# Create instances of SpinalCordTransform with desired parameters
train_transform = SpinalCordTransform(target_size=(224, 224), flip_prob=0.5)
val_transform = SpinalCordTransform(target_size=(224, 224), flip_prob=0.0)  # No flipping for validation

# Datasets
train_dataset = SpinalCordDataset(image_paths=image_paths_train, label_paths=label_paths_train, transform=train_transform)
val_dataset = SpinalCordDataset(image_paths=image_paths_val, label_paths=label_paths_val, transform=val_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

for batch in train_loader:
    images = batch["image"]  # Shape: (batch_size, 20, 3, 96, 96)
    labels = batch["label"]  # Shape: (batch_size, 20, 1, 96, 96)
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    break

for batch in val_loader:
    images = batch["image"]  # Shape: (batch_size, 20, 3, 96, 96)
    labels = batch["label"]  # Shape: (batch_size, 20, 1, 96, 96)
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    break


for idx, batch_data in enumerate(train_loader):
    if isinstance(batch_data, list):
        data, target = batch_data
    else:
        data, target = batch_data["image"], batch_data["label"]
    # data, target = data.cuda(), target.cuda()
    print(data.shape)
    data = data.permute(4, 0, 1, 2, 3).reshape(-1, 1, 224, 224)
    data = data.repeat(1, 3, 1, 1)
    target = target.permute(4, 0, 1, 2, 3).reshape(-1, 1, 224, 224)
    # target = target.repeat(1, 3, 1, 1)
    print(data.shape)
    print(target.shape)

    count = 0
    featup_list = []
    label_list = []
    for i in range(0,10,1):
        # print(i)
        # print(i*2)
        # print((i+1)*2)
        input_data = data[i*2:(i+1)*2]
        input_labels = target[i*2:(i+1)*2]
        # input_data = data[i].reshape(1,data.shape[1], data.shape[2], data.shape[3])
        # print(input_data.shape)
        input_data = input_data.cuda()
        hr_feats = upsampler(input_data)

        # print(hr_feats.shape)
        # print(input_labels.shape)

        hr_feats = hr_feats.detach().cpu().numpy()

        featup_list.append(hr_feats)
        label_list.append(input_labels)

    featup_list = np.array(featup_list)
    label_list = np.array(label_list)
    print(save_folder_train)
    print(idx)
    np.savez(save_folder_train+"/featup_imgs_"+str(idx)+".npz", featup_list)
    np.savez(save_folder_train+"/labels_"+str(idx)+".npz", label_list)


for idx, batch_data in enumerate(val_loader):
    if isinstance(batch_data, list):
        data, target = batch_data
    else:
        data, target = batch_data["image"], batch_data["label"]
    # data, target = data.cuda(), target.cuda()
    print(data.shape)
    data = data.permute(4, 0, 1, 2, 3).reshape(-1, 1, 224, 224)
    data = data.repeat(1, 3, 1, 1)
    target = target.permute(4, 0, 1, 2, 3).reshape(-1, 1, 224, 224)
    # target = target.repeat(1, 3, 1, 1)
    print(data.shape)
    print(target.shape)

    count = 0
    featup_list = []
    label_list = []
    for i in range(0,10,1):
        # print(i)
        # print(i*2)
        # print((i+1)*2)
        input_data = data[i*2:(i+1)*2]
        input_labels = target[i*2:(i+1)*2]
        # input_data = data[i].reshape(1,data.shape[1], data.shape[2], data.shape[3])
        # print(input_data.shape)
        input_data = input_data.cuda()
        hr_feats = upsampler(input_data)

        # print(hr_feats.shape)
        # print(input_labels.shape)

        hr_feats = hr_feats.detach().cpu().numpy()

        featup_list.append(hr_feats)
        label_list.append(input_labels)

    featup_list = np.array(featup_list)
    label_list = np.array(label_list)
    print(save_folder_val)
    print(idx)
    np.savez(save_folder_val+"/featup_imgs_"+str(idx)+".npz", featup_list)
    np.savez(save_folder_val+"/labels_"+str(idx)+".npz", label_list)




'''
image_nifti = []
label_nifti = []
for sub in subs:
    sessions = sorted([ses for ses in os.listdir(folder_test+sub+'/') if "ses-" in ses])
    for ses in sessions:
        file_image = glob(folder_test+sub+'/'+ses+'/anat/*_acq-lumbarMEGRE3D_desc-crop_T2starw.nii')
        file_label = glob(folder_test+sub+'/'+ses+'/anat/*_acq-lumbarMEGRE3D_desc-crop_seg-manual_label-SC_mask.nii')
        # print(file_image[0])
        if len(file_label)==0:
            print("this subject is empty "+sub)
            pass
        else:
            print(file_image[0])
            # print(file)
            print(file_image[0].split('/')[-1].split("_")[0])
            if file_image[0].split('/')[-1].split("_")[0] != sub:
                print("error with "+sub)
            else:
                image_nifti.append(file_image[0])
                label_nifti.append(file_label[0])

test_transform = SpinalCordTransform(target_size=(224, 224), flip_prob=0.0)
test_dataset = SpinalCordDataset(image_paths=image_nifti, label_paths=label_nifti, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

for batch in test_loader:
    images = batch["image"]  # Shape: (batch_size, 20, 3, 96, 96)
    labels = batch["label"]  # Shape: (batch_size, 20, 1, 96, 96)
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    break

for idx, batch_data in enumerate(test_loader):
    if isinstance(batch_data, list):
        data, target = batch_data
    else:
        data, target = batch_data["image"], batch_data["label"]
    # data, target = data.cuda(), target.cuda()
    print(data.shape)
    data = data.permute(4, 0, 1, 2, 3).reshape(-1, 1, 224, 224)
    data = data.repeat(1, 3, 1, 1)
    target = target.permute(4, 0, 1, 2, 3).reshape(-1, 1, 224, 224)
    # target = target.repeat(1, 3, 1, 1)
    print(data.shape)
    print(target.shape)

    count = 0
    featup_list = []
    label_list = []
    for i in range(0,10,1):
        # print(i)
        # print(i*2)
        # print((i+1)*2)
        input_data = data[i*2:(i+1)*2]
        input_labels = target[i*2:(i+1)*2]
        # input_data = data[i].reshape(1,data.shape[1], data.shape[2], data.shape[3])
        # print(input_data.shape)
        input_data = input_data.cuda()
        # hr_feats = upsampler(input_data)
        lr_feats = upsampler.model(input_data)

        # print(hr_feats.shape)
        # print(input_labels.shape)

        # hr_feats = hr_feats.detach().cpu().numpy()
        lr_feats = lr_feats.detach().cpu().numpy()

        # featup_list.append(hr_feats)
        featup_list.append(lr_feats)
        label_list.append(input_labels)

    featup_list = np.array(featup_list)
    label_list = np.array(label_list)
    print(save_folder_test)
    print(idx)
    np.savez(save_folder_test+"/lr_featup_imgs_"+str(idx)+".npz", featup_list)
    np.savez(save_folder_test+"/labels_"+str(idx)+".npz", label_list)

'''
