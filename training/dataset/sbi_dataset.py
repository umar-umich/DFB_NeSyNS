'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2024-01-26

The code is designed for self-blending method (SBI, CVPR 2024).
'''

import sys
sys.path.append('.')

import cv2
import yaml
import torch
import numpy as np
from copy import deepcopy
import albumentations as A
from training.dataset.albu import IsotropicResize
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.sbi_api import SBI_API


class SBIDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real lists
        # Fix the label of real images to be 0
        self.real_imglist = [(img, label) for img, label in zip(self.image_list, self.label_list) if label == 0]

        # Init SBI
        self.sbi = SBI_API(phase=mode,image_size=config['resolution'])

        # Init data augmentation method
        self.transform = self.init_data_aug_method()

    def __getitem__(self, index):
        # Get the real image paths and labels
        real_image_path, real_label = self.real_imglist[index]

        # Get the landmark paths for real images.
        #
        # The directory is configurable because THIS REPO HAS TWO landmark sets and SBI works with
        # exactly one of them. `landmarks/` holds the 5-point RetinaFace output of our v2
        # preprocessing; `sbi_api` needs dlib's 81 points, since it slices `landmark[:68]` and
        # indexes 68..80 in `reorder_landmark` to build the blend hull. Pointed at the 5-point
        # files it does not fail loudly — it indexes past the end or hulls three points — so the
        # default names the set that actually works and the assert below makes a wrong one
        # unmissable. `landmarks81/` is written by preprocessing/extract_landmarks81.py.
        landmark_dir = self.config.get('landmark_dir', 'landmarks81')
        real_landmark_path = real_image_path.replace(
            '/frames/', f'/{landmark_dir}/').replace('.png', '.npy')
        landmark = self.load_landmark(real_landmark_path).astype(np.int32)
        if landmark.shape[0] < 81:
            raise ValueError(
                f"SBI needs 81-point landmarks, got {landmark.shape[0]} from "
                f"{real_landmark_path}. `landmark_dir` is '{landmark_dir}'; the 5-point "
                f"RetinaFace files in 'landmarks/' cannot build a face hull. Run "
                f"preprocessing/extract_landmarks81.py over this corpus first.")

        # Load the real images
        real_image = self.load_rgb(real_image_path)
        real_image = np.array(real_image)  # Convert to numpy array

        # Generate the corresponding SBI sample
        fake_image, real_image = self.sbi(real_image, landmark)
        if fake_image is None:
            fake_image = deepcopy(real_image)
            fake_label = 0
        else:
            fake_label = 1

        # To tensor and normalize for fake and real images
        fake_image_trans = self.normalize(self.to_tensor(fake_image))
        real_image_trans = self.normalize(self.to_tensor(real_image))

        return {"fake": (fake_image_trans, fake_label), 
                "real": (real_image_trans, real_label)}

    def __len__(self):
        return len(self.real_imglist)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor and label tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors for fake and real data
        fake_images, fake_labels = zip(*[data["fake"] for data in batch])
        real_images, real_labels = zip(*[data["real"] for data in batch])

        # Stack the image, label, landmark, and mask tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        
        data_dict = {
            'image': images,
            'label': labels,
            'landmark': None,
            'mask': None,
        }
        return data_dict

    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p = 0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            additional_targets={'real': 'sbi'},
        )
        return trans


if __name__ == '__main__':
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/sbi.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = SBIDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        if iteration > 10:
            break