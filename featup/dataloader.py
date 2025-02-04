from torch.utils.data import Dataset, DataLoader
import nibabel as nb
import torch
import numpy as np
from scipy.ndimage import zoom
### THIS IS TO ENSURE THE CHANGE

class SpinalCordDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        assert len(image_paths) == len(label_paths), "Image and label paths must have the same length"
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load the NIfTI files
        image = nb.load(image_path).get_fdata(dtype=np.float32)  # Shape: (H, W, D)
        label = nb.load(label_path).get_fdata(dtype=np.float32)  # Shape: (H, W, D)
        # Extract specific slices
        image_slices = image[:, :, :20]  # Extract slices 0-19
        label_slices = label[:, :, :20]  # Extract corresponding slices

        # Apply transforms to each slice
        transformed_images = []
        transformed_labels = []

        for i in range(image_slices.shape[2]):
            img_slice = image_slices[:, :, i]
            lbl_slice = label_slices[:, :, i]

            if self.transform:
                img_slice, lbl_slice = self.transform(img_slice, lbl_slice)

            # Convert to PyTorch tensors
            img_tensor = torch.tensor(img_slice, dtype=torch.float32).permute(2, 0, 1)  # Shape: (3, H, W)
            lbl_tensor = torch.tensor(lbl_slice, dtype=torch.float32).unsqueeze(0)      # Shape: (1, H, W)

            transformed_images.append(img_tensor)
            transformed_labels.append(lbl_tensor)

        # Stack slices to form 3D volumes
        transformed_images = torch.stack(transformed_images, dim=3)  # Shape: (20, 3, H, W)
        transformed_labels = torch.stack(transformed_labels, dim=3)  # Shape: (20, 1, H, W)

        

        return {
            "image": transformed_images,
            "label": transformed_labels
        }



class SpinalCordTransform:
    def __init__(self, target_size=(224, 224), flip_prob=0.5):
        self.target_size = target_size
        self.flip_prob = flip_prob

    def resize(self, image, label):
        """
        Resize image and label to the target size.
        """
        zoom_factors_image = (
            self.target_size[0] / image.shape[0],
            self.target_size[1] / image.shape[1]
        )
        zoom_factors_label = (
            self.target_size[0] / label.shape[0],
            self.target_size[1] / label.shape[1]
        )
        resized_image = zoom(image, zoom_factors_image, order=1)  # Bilinear interpolation
        resized_label = zoom(label, zoom_factors_label, order=0)  # Nearest-neighbor interpolation for labels
        return resized_image, resized_label

    def normalize(self, slice_2d):
        """
        Normalize intensity values to [0, 1].
        """
        slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
        return slice_2d

    def random_flip(self, image, label):
        """
        Randomly flip along spatial axes and ensure contiguous memory.
        """
        if np.random.rand() < self.flip_prob:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        if np.random.rand() < self.flip_prob:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        return image, label

    def grayscale_to_rgb(self, slice_2d):
        """
        Convert a grayscale slice to RGB by duplicating the grayscale values across three channels.
        """
        return np.stack([slice_2d] * 3, axis=-1)  # Shape: (H, W, 3)

    def __call__(self, image, label):
        """
        Apply all transformations to an image and label.
        """
        # Resize to target size
        image, label = self.resize(image, label)

        # Normalize image
        image = self.normalize(image)
        label = self.normalize(label)

        # Random flips
        image, label = self.random_flip(image, label)

        # Convert image to RGB
        # image = self.grayscale_to_rgb(image)  # Shape: (H, W, 3)
        image = image.reshape(image.shape[0], image.shape[1],1)

        return image, label