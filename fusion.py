import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19

class VGG19(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(VGG19, self).__init__()
        features = list(vgg19(pretrained=True).features)
        if device == "cuda":
            self.features = nn.ModuleList(features).cuda().eval()
        else:
            self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        feature_maps = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.modules.ReLU):
                feature_maps.append(x)
        return feature_maps

class Fusion:
    def __init__(self, input):
        """
        Class Fusion constructor

        Instance Variables:
            self.images: input images
            self.model: CNN model, default=vgg19
            self.device: either 'cuda' or 'cpu'
        """
        self.input_images = input
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VGG19(self.device)

    def fuse(self):
        """
        A top level method which fuse self.images
        """
        # Convert all images to YCbCr format
        self.YCBCr_images = []
        for img in self.input_images:
            if img.ndim == 3:
                self.YCBCr_images.append(self._RGB_to_YCbCr(img))
        # Transfer all images to PyTorch tensors
        self._tranfer_to_tensor()
        # Perform fuse strategy
        self._fuse()


    def _fuse(self):
        with torch.no_grad():
            imgs_to_sum_maps = {}
            for tensor_img in self.images_to_tensors:
                imgs_to_sum_maps[tensor_img] = []
                feature_maps = self.model(tensor_img)
                for feature_map in feature_maps:
                    sum_map = torch.sum(feature_map, dim=1, keepdim=True)
                    imgs_to_sum_maps[tensor_img].append(sum_map)



    def _RGB_to_YCbCr(self, img_RGB):
        """
        A private method which converts an RGB image to YCrCb format
        """
        img_RGB = img_RGB.astype(np.float32) / 255.
        return cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)[:, :, 0]

    def _YCbCr_to_RGB(self, img_YCbCr):
        """
        A private method which converts a YCrCb image to RGB format
        """
        img_YCbCr = img_YCbCr.astype(np.float32)
        return cv2.cvtColor(img_YCbCr, cv2.COLOR_YCrCb2RGB)

    def _tranfer_to_tensor(self):
        """
        A private method to transfer all input images to PyTorch tensors
        """
        self.images_to_tensors = []
        for image in self.input_images:
            np_input = image.astype(np.float32)
            if np_input.ndim == 2:
                np_input = np.repeat(np_input[None, None], 3, axis=1)
            else:
                np_input = np.transpose(np_input, (2, 0, 1))[None]
            if self.device == "cuda":
                self.images_to_tensors.append(torch.from_numpy(np_input).cuda())
            else:
                self.images_to_tensors.append(torch.from_numpy(np_input))


if __name__ == '__main__':
    img1 = cv2.imread('images/MRI-SPECT/mr.png')
    img2 = cv2.imread('images/MRI-SPECT/tc.png')
    FU = Fusion([img1, img2])
    fusion_img = FU.fuse()
