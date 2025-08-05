import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import pandas as pd
import timm
import math
import scipy.stats as st
from torch.utils import data
import random
from torch.nn import functional as F
from torch import nn
from robustbench.utils import load_model

img_height, img_width = 224, 224
img_max, img_min = 1., 0

special_model={'inception_v4':'/home/user/.cache/torch/hub/checkpoints/inceptionv4-8e4777a0.pth','inception_resnet_v2':'/home/user/.cache/torch/hub/checkpoints/inception_resnet_v2-940b1cd6.pth'}

cnn_model_paper = ['resnet18','resnet101','resnext50_32x4d', 'inception_v3','vgg16']
vit_model_paper = ['vit_base_patch16_224', 'pit_b_224',
                   'visformer_small', 'swin_tiny_patch4_window7_224','inception_v3.tf_adv_in1k',' inception_resnet_v2.tf_ens_adv_in1k','convnext_base','edgenext_base','convit_base','inception_v4','inception_resnet_v2']

robustbenchs=['Amini2024MeanSparse','Chen2024Data_WRN_50_2','Liu2023Comprehensive_ConvNeXt-B']

cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101',
                 'resnext50_32x4d', 'densenet121', 'mobilenet_v2']
vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                 'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

tgr_vit_model_list = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                      'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base']


def load_pretrained_model(cnn_model=[], vit_model=[],robustbenchs=[]):
    for model_name in cnn_model:
        yield model_name, models.__dict__[model_name](weights="DEFAULT")
        # yield model_name, models.__dict__[model_name](weights="IMAGENET1K_V1")
    for model_name in vit_model:
        # print(timm.list_models())
        if model_name in special_model.keys():
            # model = timm.create_model(model_name, pretrained=True,pretrained_cfg_overlay=dict(file=special_model[model_name]))
            model = timm.create_model(model_name, pretrained=True)
        else:
            model = timm.create_model(model_name, pretrained=True)
        yield model_name, model
    for model_name in robustbenchs:
        yield model_name, load_model(model_name,dataset='imagenet',threat_model='Linf')


def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)


def save_images(output_dir, adversaries, filenames,png=True):
    adversaries = (adversaries.detach().permute((0,2,3,1)).detach().cpu().numpy() * 255).astype(np.uint8)
    os.makedirs(output_dir, exist_ok=True)
    for i, filename in enumerate(filenames):
        if png:
            Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename.replace("JPEG","png")))
        else:
            Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, eval=False,png=False):
        self.targeted = targeted
        self.data_dir = input_dir
        self.eval=eval
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))
        self.png=png

        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/labels.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        if self.eval:
            if not self.png:
                filepath = os.path.join(self.data_dir, filename)
                image = Image.open(filepath)
            else:
                filepath = os.path.join(self.data_dir, filename.replace("JPEG","png"))
                image = Image.open(filepath)
        else:
            filepath = os.path.join(self.data_dir, filename)
            image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'],
                                             dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l

class NIPS_GAME(data.Dataset):
    def __init__(self, dir, csv_path, transforms=None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path, engine='python')
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + ".png"
        
        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = pil_img
        return data, ImageID, Truelabel, TargetClass

    def __len__(self):
        return len(self.csv)

def seed_torch(seed):
    """Set a random seed to ensure that the results are reproducible"""  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def save_img(save_path, img, split_channel=False):
    img_ = np.array(img * 255).astype('uint8')
    if split_channel:
        for i in range(img_.shape[2]):
            ch_path = save_path + "@channel{}.jpg".format(i)
            ch = Image.fromarray(img_[:, :, i])
            ch.save(ch_path)
    else:
        Image.fromarray(img_).save(save_path)

class ANDA:
    def __init__(self, device, data_shape=(1, 3, 299, 299)):
        self.data_shape = data_shape
        self.device = device

        self.n_models = 0
        self.noise_mean = torch.zeros(data_shape, dtype=torch.float).to(device)
        self.noise_cov_mat_sqrt = torch.empty((0, np.prod(data_shape)), dtype=torch.float).to(device)

    def sample(self, n_sample=1, scale=0.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        mean = self.noise_mean
        cov_mat_sqrt = self.noise_cov_mat_sqrt

        if scale == 0.0:
            assert n_sample == 1
            return mean.unsqueeze(0)

        assert scale == 1.0
        k = cov_mat_sqrt.shape[0]
        cov_sample = cov_mat_sqrt.new_empty((n_sample, k), requires_grad=False).normal_().matmul(cov_mat_sqrt)
        cov_sample /= (k - 1)**0.5

        rand_sample = cov_sample.reshape(n_sample, *self.data_shape)
        sample = mean.unsqueeze(0) + scale * rand_sample
        sample = sample.reshape(n_sample, *self.data_shape)
        return sample

    def collect_model(self, noise):
        mean = self.noise_mean
        cov_mat_sqrt = self.noise_cov_mat_sqrt
        assert noise.device == cov_mat_sqrt.device
        bs = noise.shape[0]
        # first moment
        mean = mean * self.n_models / (self.n_models + bs) + noise.data.sum(dim=0, keepdim=True) / (self.n_models + bs)

        # square root of covariance matrix
        dev = (noise.data - mean).view(bs, -1)
        cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev), dim=0)

        self.noise_mean = mean
        self.noise_cov_mat_sqrt = cov_mat_sqrt
        self.n_models += bs

    def clear(self):
        self.n_models = 0
        self.noise_mean = torch.zeros(self.data_shape, dtype=torch.float).to(self.device)
        self.noise_cov_mat_sqrt = torch.empty((0, np.prod(self.data_shape)), dtype=torch.float).to(self.device)
        
 
def is_sqr(n):
    a = int(math.sqrt(n))
    return a * a == n

def get_theta(i, j):
    theta = torch.tensor([[[1, 0, i], [0, 1, j]]], dtype=torch.float)
    return theta

def get_thetas(n, min_r=-0.5, max_r=0.5):
    range_r = torch.linspace(min_r, max_r, n)
    thetas = []
    for i in range_r:
        for j in range_r:
            thetas.append(get_theta(i, j))
    thetas = torch.cat(thetas, dim=0)
    return thetas

def translation(thetas, imgs):
    grids = F.affine_grid(thetas, imgs.size(), align_corners=False).to(imgs.device)
    output = F.grid_sample(imgs, grids, align_corners=False)
    return output

def scale_transform(input_tensor, m=5):
    outs = [(input_tensor) / (2**i) for i in range(m)]
    x_batch = torch.cat(outs, dim=0)
    return x_batch

# def scale_transform(input_tensor, m=5):
#     shape = input_tensor.shape
#     outs = [(input_tensor) / (2**i) for i in range(m)]
#     x_batch = torch.cat(outs, dim=0)
#     new_shape = x_batch.shape
#     x_batch = x_batch.reshape(m, *shape).transpose(1, 0).reshape(*new_shape)
#     return x_batch

class Translation_Kernel:
    def __init__(self, len_kernel=15, nsig=3, kernel_name='gaussian'):
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
        
def input_diversity(input_tensor, resize=330, diversity_prob=0.5):
    if torch.rand(1) >= diversity_prob:
        return input_tensor
    image_width = input_tensor.shape[-1]
    assert image_width == 299, "only support ImageNet"
    rnd = torch.randint(image_width, resize, ())
    rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='bilinear', align_corners=True)
    h_rem = resize - rnd
    w_rem = resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    return padded


def get_minibatch(x: torch.Tensor, y: torch.Tensor, minibatch: int):
    nsize = x.shape[0]
    start = 0
    while start < nsize:
        end = min(nsize, start + minibatch)
        yield x[start:end], y[start:end]
        start += minibatch


if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted',
                         targeted=True, eval=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break
