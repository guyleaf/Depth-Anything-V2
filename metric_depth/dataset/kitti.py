import os

import cv2
import torch
from dataset.transform import NormalizeImage, PrepareForNet, Resize
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class KITTI(Dataset):
    def __init__(self, root_dir, filelist_path, mode, size=(518, 518)):
        if mode != 'val':
            raise NotImplementedError
        
        self.mode = mode
        self.size = size
        
        self.root_dir = root_dir
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):
        img_path, depth_path = map(lambda path: os.path.join(self.root_dir, path), self.filelist[item].split(' '))
        # img_path = self.filelist[item].split(' ')[0]
        # depth_path = self.filelist[item].split(' ')[1]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')
        
        sample = self.transform({'image': image, 'depth': depth})
        
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['depth'] = sample['depth'] / 256.0  # convert in meters
        
        sample['valid_mask'] = sample['depth'] > 0
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)
