from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
from .fourier_perlin_noise import fourier_perlin_noise
import cv2

class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0, 1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

def get_data_transforms(size):
    data_transforms = transforms.Compose([Normalize(), ToTensor()])
    gt_transforms = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTecDataset_train(torch.utils.data.Dataset):
    def __init__(self, root, transform, type_dataset, dtd_paths, to_memory_normal=False, to_memory_abnormal=False):
        self.img_path = root
        self.type_dataset = type_dataset
        self.transform = transform
        self.dtd_paths = dtd_paths
        self.img_paths, self.additional_img_paths = self.load_dataset() # good: 0, anomaly: 1
        self.to_memory_normal = to_memory_normal
        self.to_memory_abnormal = to_memory_abnormal
        self.anomaly_switch = False

    def load_dataset(self):
        if self.type_dataset == 'mvtec':
            img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.png")
        elif self.type_dataset == 'btad':
            if self.img_path.split('/')[2] == '02':
                img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.png")
            else:
                img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.bmp")
        elif self.type_dataset == 'visa':
            img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.JPG")
        additional_img_paths = glob.glob(self.dtd_paths + "/*/*.jpg")
        return img_paths, additional_img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., (256, 256)) # [256, 256, 3]
        # img_normal = self.transform(img) # [3, 256, 256]
        ### perlin noise + fourier mixup
        # src_img = Image.open(img_path).convert('RGB').resize((256, 256), Image.BILINEAR)
        # dest_index = torch.randint(0, len(self.additional_img_paths), (1,)).item()
        # dest_img = Image.open(self.additional_img_paths[dest_index]).convert('RGB').resize((256, 256), Image.BILINEAR)
        # img_noise = fourier_perlin_noise(src_img, dest_img, 1.0)
        # img_noise = self.transform(img_noise)
        # # anomaly switch
        # if self.to_memory_normal and not self.to_memory_abnormal:
        #     return img_normal
        # elif not self.to_memory_normal and self.to_memory_abnormal:
        #     return img_noise
        # elif not self.to_memory_normal and not self.to_memory_abnormal:
        #     if self.anomaly_switch:
        #         img_normal = img_noise
        #         self.anomaly_switch = False
        #     else:
        #         self.anomaly_switch = True
        #     return img_normal
        if self.to_memory_normal and not self.to_memory_abnormal: # for normal memory module
            img = self.transform(img)
            return img
        elif not self.to_memory_normal and self.to_memory_abnormal: # for abnormal memory module
            img_noise = self.synthesis_anomaly(img_path)
            img_noise = self.transform(img_noise)
            return img_noise
        elif not self.to_memory_normal and not self.to_memory_abnormal: # for training
            if self.anomaly_switch:
                img = self.synthesis_anomaly(img_path)
                self.anomaly_switch = False
            else:
                self.anomaly_switch = True
            img = self.transform(img)
            return img

    def synthesis_anomaly(self, img_path):
        src_img = Image.open(img_path).convert('RGB').resize((256, 256), Image.BILINEAR)
        dest_index = torch.randint(0, len(self.additional_img_paths), (1,)).item()
        dest_img = Image.open(self.additional_img_paths[dest_index]).convert('RGB').resize((256, 256), Image.BILINEAR)
        img_noise = fourier_perlin_noise(src_img, dest_img, 1.0)
        return img_noise

class MVTecDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, type_dataset):
        self.type_dataset = type_dataset
        self.img_path = os.path.join(root, 'test')
        self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # good: 0, anomaly: 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)
        for defect_type in defect_types:
            if defect_type == 'good':
                if self.type_dataset == 'mvtec':
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                elif self.type_dataset == 'btad':
                    img_path_split = self.img_path.split('/')[-2]
                    if img_path_split == '02':
                        img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                    else:
                        img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                elif self.type_dataset == 'visa':
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.type_dataset == 'mvtec':
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                elif self.type_dataset == 'btad':
                    img_path_split = self.img_path.split('/')[-2]
                    if img_path_split == '01':
                        img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                        gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                    elif img_path_split == '02':
                        img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                        gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                    else:
                        img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                        gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.bmp")
                elif self.type_dataset == 'visa':
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., (256, 256))
        img = self.transform(img)  # [3, 256, 256]
        if gt == 0: # good
            gt = torch.zeros([1, img.shape[-1], img.shape[-1]])
        else: # anomaly
            gt = Image.open(gt)
            gt = self.gt_transform(gt)  # [1, 256, 256]
        assert img.shape[1:] == gt.shape[1:], "image.size != gt.size"
        return img, gt.squeeze(0), label