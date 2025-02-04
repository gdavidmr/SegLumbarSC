import torch
import torchvision.transforms as T
from PIL import Image
from dataloader import SpinalCordDataset, SpinalCordTransform
from model import SegmentationCNN
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial
from trainer_unet import run_training
from argparse import Namespace
import sys
import numpy as np
import matplotlib.pyplot as plt

# sys.path.append('/home/sarvagya-pc/Desktop/FeatUp/featup')

from featup.util import norm, unnorm
from featup.plotting import plot_feats, plot_lang_heatmaps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_norm = True
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=use_norm).to(device)

save_folder_test = "/home/sarvagya-pc/Desktop/Balgrist_neuroimg/featup/data/hr/sub_wise"

sub = "sub-81"
ses = "ses-01"

# image_nifti = ["/home/sarvagya-pc/Desktop/Balgrist_neuroimg/for_test/"+sub+"/"+ses+"/anat/"+sub+"_"+ses+"_acq-lumbarMEGRE3D_desc-crop_T2starw.nii"]
# image_nifti = ["/media/sarvagya-pc/2TB HDD/Balgrist/Segm_extra/Cervical_MEGRE/sub-9024/output_cr.nii"]
image_nifti = ["/home/sarvagya-pc/Desktop/Balgrist_neuroimg/for_test/output_file.nii"]
label_nifti = ["/home/sarvagya-pc/Desktop/Balgrist_neuroimg/for_test/"+sub+"/"+ses+"/anat/"+sub+"_"+ses+"_acq-lumbarMEGRE3D_desc-crop_seg-manual_label-SC_mask.nii"]

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
        hr_feats = upsampler(input_data)

        # print(hr_feats.shape)
        # print(input_labels.shape)

        hr_feats = hr_feats.detach().cpu().numpy()

        featup_list.append(hr_feats)
        label_list.append(input_labels)

    featup_list = np.array(featup_list)
    label_list = np.array(label_list)
    print(save_folder_test)
    print(idx)
    np.savez(save_folder_test+"/featup_imgs_t2w"+sub+"_"+ses+".npz", featup_list)
    np.savez(save_folder_test+"/labels_cervical"+sub+"_"+ses+".npz", label_list)