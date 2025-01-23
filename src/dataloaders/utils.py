import os
import os.path
import hashlib
import errno
import torch
from torchvision import transforms
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from torchvision import transforms as T
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import torch
from torch.utils.data import random_split, DataLoader, Dataset, Subset

dataset_stats = {
    'CIFAR100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
                 'size' : 32}, 
    'ImageNet_R': {
                 'size' : 224}, 
    'DomainNet': {
                 'size' : 224},  
                }
                
# transformations
def get_transform(dataset='cifar100', phase='test', aug=True, resize_imnet=False):
    transform_list = []
    # get out size
    crop_size = dataset_stats[dataset]['size']

    # get mean and std
    dset_mean = (0.0,0.0,0.0) # dataset_stats[dataset]['mean']
    dset_std = (1.0,1.0,1.0) # dataset_stats[dataset]['std']

    if dataset == 'ImageNet32' or dataset == 'ImageNet84':
        transform_list.extend([
            transforms.Resize((crop_size,crop_size))
        ])

    if phase == 'train':
        transform_list.extend([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dset_mean, dset_std),
                            ])
    else:
        if dataset.startswith('ImageNet') or dataset == 'DomainNet':
            transform_list.extend([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
        else:
            transform_list.extend([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])


    return transforms.Compose(transform_list)

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)

def accuracy(output, target, topk=(1,)):
    ### 返回的是正确的个数
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return [correct[:min(k, maxk)].reshape(-1).float().sum(0)  for k in topk],batch_size


def global_distillation_loss(output,outputs):
    mse = torch.nn.MSELoss(reduction='sum')
    total_loss = 0
    for i in outputs:
        loss=mse(output,i)
        total_loss +=loss
    return total_loss

def build_transform(is_train,input_size):
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

def getDataloader(client_dataset, batch_size, client_index, task_id):
    train_dataset = client_dataset[client_index][task_id]

    train_loader = DataLoader(dataset=train_dataset,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  pin_memory=True)

    return train_loader

def split_dataset_by_target(dataset, class_real):
    # 创建一个字典，用于存储每个类的索引
    class_indices = {i: [] for i in class_real}

    # print("class_real", class_real)
    # for idx, (_, target) in enumerate(dataset):
    #     print(target, end=" ")
    # 遍历数据集，收集每个类的索引
    for idx, (_, target) in enumerate(dataset):
        class_indices[target].append(idx)

    # 创建子数据集列表
    subsets = {}
    for class_id, indices in class_indices.items():
        subsets[class_id] = Subset(dataset, indices)

    return subsets