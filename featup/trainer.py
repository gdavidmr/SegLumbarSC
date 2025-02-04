import torch
import torchvision.transforms as T
from PIL import Image

# from featup.util import norm, unnorm
# from featup.plotting import plot_feats, plot_lang_heatmaps
from dataloader_training import SpinalCordDataset, SpinalCordTransform
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
from tensorboardX import SummaryWriter
from trainer_unet import run_training
from argparse import Namespace
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

args = Namespace(
    checkpoint=None,
    logdir="test_GM",
    pretrained_dir="/home/sarvagya-pc/Desktop/Balgrist_neuroimg/UNETR/pretrained_model/",
    data_dir="/dataset/dataset0/",
    json_list="dataset_0.json",
    pretrained_model_name="UNETR_model_best_acc.pth",
    save_checkpoint=True,  # Set to True if you want to save checkpoints
    max_epochs=500,
    batch_size=1,
    sw_batch_size=1,
    optim_lr=1e-4,
    optim_name="adamw",
    reg_weight=1e-5,
    momentum=0.99,
    noamp=False,  # Set to True if you don't want AMP (Automatic Mixed Precision)
    val_every=2,
    distributed=False,  # Set to True for distributed training
    world_size=1,
    rank=0,
    dist_url="tcp://127.0.0.1:23456",
    dist_backend="nccl",
    workers=8,
    model_name="unetr",
    pos_embed="perceptron",
    norm_name="instance",
    num_heads=12,
    mlp_dim=3072,
    hidden_size=768,
    feature_size=16,
    in_channels=1,
    out_channels=1,
    res_block=False,  # Set to True if using residual blocks
    conv_block=False,  # Set to True if using convolutional blocks
    use_normal_dataset=False,  # Set to True if using MONAI's Dataset class
    a_min=-175.0,
    a_max=250.0,
    b_min=0.0,
    b_max=1.0,
    space_x=1.5,
    space_y=1.5,
    space_z=2.0,
    roi_x=96,
    roi_y=96,
    roi_z=3,
    dropout_rate=0.0,
    RandFlipd_prob=0.2,
    RandRotate90d_prob=0.2,
    RandScaleIntensityd_prob=0.1,
    RandShiftIntensityd_prob=0.1,
    infer_overlap=0.5,
    lrschedule="warmup_cosine",
    warmup_epochs=50,
    resume_ckpt=False,
    resume_jit=False,
    smooth_dr=1e-6,
    smooth_nr=0.0
)

args.amp = not args.noamp



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_norm = True
# upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=use_norm).to(device)
model = SegmentationCNN().to(device)

# Load the checkpoint
checkpoint = torch.load("/home/sarvagya-pc/Desktop/FeatUp/test/model_final_first_500_epochs.pt")

# Adjust the output layer weights
state_dict = checkpoint["state_dict"]

# Load the remaining weights into the model
model.load_state_dict(state_dict, strict=False)


folder = "/home/sarvagya-pc/Desktop/Balgrist_neuroimg/featup/data/GM/hr/"

image_npz_train = sorted(glob(folder+'/train/featup_imgs_*.npz'))
label_npz_train = sorted(glob(folder+'/train/labels_*.npz'))

image_npz_val = sorted(glob(folder+'/val/featup_imgs_*.npz'))
label_npz_val = sorted(glob(folder+'/val/labels_*.npz'))

# if len(file_label)==0:
#     print("this subject is empty ")
#     pass
# else:
#     print(file_image[0])
#     # print(file)
#     print(file_image[0].split('/')[-1].split("_")[0])
#     image_npz.append(file_image[0])
#     label_npz.append(file_label[0])

# print(image_npz)
# print("***********************************************************************")
# print(label_npz)


# X_train, X_test, y_train, y_test = train_test_split(image_nifti, label_nifti, test_size=0.2, random_state=1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


# image_paths_train = X_train
# label_paths_train = y_train

# image_paths_val = X_val
# label_paths_val = y_val

# # Create Dataset and DataLoader
# transform = SpinalCordTransform(flip_prob=0.5)
# dataset = SpinalCordDataset(image_paths=image_paths, label_paths=label_paths, transform=transform)

# data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# Create instances of SpinalCordTransform with desired parameters
train_transform = SpinalCordTransform(target_size=(224, 224), flip_prob=0.5)
val_transform = SpinalCordTransform(target_size=(224, 224), flip_prob=0.0)  # No flipping for validation

# Datasets
train_dataset = SpinalCordDataset(image_paths=image_npz_train, label_paths=label_npz_train, transform=train_transform)
val_dataset = SpinalCordDataset(image_paths=image_npz_val, label_paths=label_npz_val, transform=val_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

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


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
dice_loss = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6)
dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
# model_inferer = partial(
#     sliding_window_inference,
#     roi_size=[96,96,20],
#     sw_batch_size=1,
#     predictor=model,
#     overlap=0.5,
# )
scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=50, max_epochs=5000
        )
start_epoch = 0
# post_label = AsDiscrete(to_onehot=False, n_classes=args.out_channels)
# post_pred = AsDiscrete(argmax=True, to_onehot=False, n_classes=args.out_channels)

# Ground truth labels: no transformation needed
post_label = AsDiscrete()  # Use as-is if labels are already binary

# Model predictions: threshold to binarize probabilities
post_pred = AsDiscrete(threshold_values=0.5)

accuracy = run_training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_func=dice_loss,
    args=args,
    acc_func=dice_acc,
    scheduler=scheduler,
    start_epoch=start_epoch,
    post_label=post_label,
    post_pred=post_pred
)
