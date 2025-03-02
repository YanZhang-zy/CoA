import os
import random
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, Resize
from torchvision.transforms import functional as FF


def preprocess_feature(img):
    img = ToTensor()(img)
    clip_normalizer = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img = clip_normalizer(img)
    return img


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size=256, format='.png'):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 100)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        split_name = os.path.split(img)[-1].split('_')
        id = split_name[0]
        clear_name = id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            target = RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.haze_imgs)


class RESIDE_Dataset_2(data.Dataset):
    def __init__(self, path, train, size=256, format='.jpg'):
        super(RESIDE_Dataset_2, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 100)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        split_name = os.path.split(img)[-1]


        id = os.path.splitext(split_name)[0]


        clear_name = f"{id}{self.format}"
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            target = RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.haze_imgs)


class TestDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()
        self.clear_image_list.sort()
        self.size = 256

    def __getitem__(self, index):
        # data shape: C*H*W
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = hazy_image_name.split('_')[0] + '.png'

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(hazy, output_size=(self.size, self.size))
            hazy = FF.crop(hazy, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        hazy, clear = self.augData(hazy.convert("RGB"), clear.convert("RGB"))

        return hazy, clear, hazy_image_name

    def augData(self, data, target):
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.hazy_image_list)




class CLIP_loader(data.Dataset):

    def __init__(self, hazy_path, train, size=256):
        self.hazy_path = hazy_path
        self.train = train
        self.hazy_image_list = os.listdir(hazy_path)
        self.hazy_image_list.sort()
        self.size = size

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        hazy = Image.open(hazy_image_path).convert('RGB')
        width, height = hazy.size
        crop_size = min(self.size, height, width)

        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(hazy, output_size=(crop_size, crop_size))
            hazy = FF.crop(hazy, i, j, h, w)
        hazy = Resize((self.size, self.size))(hazy)
        hazy = self.augData(hazy.convert("RGB"))
        return hazy

    def augData(self, data):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
        return preprocess_feature(data)

    def __len__(self):
        return len(self.hazy_image_list)
