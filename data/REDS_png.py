import os
import random
from os.path import join

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import normalize, Crop, Flip, ToTensor


class SuperRDataset(Dataset):
    """
    Structure of self_.records:
        seq:
            frame:
                path of images -> {'Blur': <path>, 'Sharp': <path>}
    """

    def __init__(self, path, ds_type, frames, future_frames, past_frames, scale = 4, crop_size=(256, 256), data_format='RGB',
                 centralize=True, normalize=True):
        assert frames - future_frames - past_frames >= 1
        data_format='RGB'
        self.frames = frames
        self.num_ff = future_frames
        self.num_pf = past_frames
        self.data_format = data_format
        self.scale = scale
        self.W, self.H = 1280, 720
        self.lrW, self.lrH = 320, 180
        #print(crop_size)
        self.crop_h, self.crop_w = 256, 256#crop_size
        self.normalize = normalize
        self.centralize = centralize
        self.transform = transforms.Compose([Crop((self.crop_h, self.crop_w)), Flip(), ToTensor()])
        self._seq_length = 100
        self._samples = self._generate_samples(path, ds_type)

    def _generate_samples(self, dataset_path, ds_type):
        samples = list()
        records = dict()
        data_format = 'RGB'
        #should be the sharp dataset path
        #e.g. , '.../train/train_sharp/X4'
        seqs_path = join(dataset_path, ds_type, ds_type+'_sharp')
        #seqs = sorted(os.listdir(dataset_path), key=int)
        seqs = sorted(os.listdir(seqs_path), key=int)
        #suffix = 'png'
        #print(join(dataset_path, ds_type, ds_type+'_sharp_bicubic', 'X4', '{:08d}.{}'.format(2, suffix)))
        for seq in seqs:
            records[seq] = list()
            for frame in range(self._seq_length):
                suffix = 'png' if data_format == 'RGB' else 'tiff'
                sample = dict()
                sample['Bicub'] = join(dataset_path, ds_type, ds_type+'_sharp_bicubic', 'X4', seq, '{:08d}.{}'.format(frame, suffix))
                sample['Sharp'] = join(dataset_path, ds_type, ds_type+'_sharp', seq, '{:08d}.{}'.format(frame, suffix))
                records[seq].append(sample)
        for seq_records in records.values():
            temp_length = len(seq_records) - (self.frames - 1)
            if temp_length <= 0:
                raise IndexError('Exceed the maximum length of the video sequence')
            for idx in range(temp_length):
                samples.append(seq_records[idx:idx + self.frames])
        return samples

    def __getitem__(self, item):
        top = random.randint(0, self.H - self.crop_h)
        left = random.randint(0, self.W - self.crop_w)
        flip_lr = random.randint(0, 1)
        flip_ud = random.randint(0, 1)
        sample = {'top': top, 'left': left, 'flip_lr': flip_lr, 'flip_ud': flip_ud}

        lr_imgs, hr_imgs = [], []
        for sample_dict in self._samples[item]:
            lr_img, hr_img = self._load_sample(sample_dict, sample)
            lr_imgs.append(lr_img)
            hr_imgs.append(hr_img)
        hr_imgs = hr_imgs[self.num_pf:self.frames - self.num_ff]
        return [torch.cat(item, dim=0) for item in [lr_imgs, hr_imgs]]

    def _load_sample(self, sample_dict, sample):
        if self.data_format == 'RGB':
            gt_img = cv2.imread(sample_dict['Sharp'])
            lr_img = cv2.imread(sample_dict['Bicub'])
            sample['image'] = lr_img
            sample['label'] = gt_img
        elif self.data_format == 'RAW':
            sample['image'] = cv2.imread(sample_dict['Blur'], -1)[..., np.newaxis].astype(np.int32)
            sample['label'] = cv2.imread(sample_dict['Sharp'], -1)[..., np.newaxis].astype(np.int32)
        sample = self.transform(sample)
        val_range = 2.0 ** 8 - 1 if self.data_format == 'RGB' else 2.0 ** 16 - 1
        lr_img = normalize(sample['image'], centralize=self.centralize, normalize=self.normalize, val_range=val_range)
        gt_img = normalize(sample['label'], centralize=self.centralize, normalize=self.normalize, val_range=val_range)
        #print(lr_img.shape, gt_img.shape)
        return lr_img, gt_img

    def __len__(self):
        return len(self._samples)


class Dataloader:
    def __init__(self, para, device_id, ds_type='train'):
        #path = join(para.data_root, para.dataset, '{}_{}'.format(para.dataset, para.ds_config), ds_type)
        path = join(para.data_root, para.dataset)
        frames = para.frames
        #dataset = DeblurDataset(path, frames, para.future_frames, para.past_frames, para.patch_size, para.data_format,
        #                        para.centralize, para.normalize)
        dataset = SuperRDataset(path, ds_type, frames, para.future_frames, para.past_frames, para.patch_size, para.data_format,
                                para.centralize, para.normalize)
        gpus = para.num_gpus
        bs = para.batch_size
        ds_len = len(dataset)
        if para.trainer_mode == 'ddp':
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=para.num_gpus,
                rank=device_id
            )
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=False,
                num_workers=para.threads,
                pin_memory=True,
                sampler=sampler,
                drop_last=True
            )
            loader_len = np.ceil(ds_len / gpus)
            self.loader_len = int(np.ceil(loader_len / bs) * bs)

        elif para.trainer_mode == 'dp':
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=True,
                num_workers=para.threads,
                pin_memory=True,
                drop_last=True
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len


if __name__ == '__main__':
    from para import Parameter

    para = Parameter().args
    para.data_format = 'RAW'
    para.dataset = 'BSD'
    dataloader = Dataloader(para, 0)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break
    print(x.type(), y.type())
    print(np.max(x.numpy()), np.min(x.numpy()))
    print(np.max(y.numpy()), np.min(y.numpy()))
