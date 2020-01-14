import cv2
import numpy as np
import torch

from torchvision.models.vgg import vgg19

class Fusion:
    def __init__(self, input, model=None):
        """
        Class Fusion constructor

        Instance Variables:
            self.images: input images
            self.model: CNN model, default=vgg19
            self.device: either 'cuda' or 'cpu'
        """
        self.input_images = input
        self.model = vgg19(True) if model is None else model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.model.cuda().eval()
        elif self.device == "cpu":
            self.model.eval()
        self.conv2d_layers = [
            p for _, p in self.model.features._modules.items()
            if isinstance(p, torch.nn.modules.conv.Conv2d)]
        self.relu_layers = [
            p for _, p in self.model.features._modules.items()
            if isinstance(p, torch.nn.modules.ReLU)]

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
        output = self._fuse()

    def _fuse(self):
        with torch.no_grad():
            pass

    def _RGB_to_YCbCr(self, img_rgb):
        """
        A private method which converts an RGB image to YCrCb format
        """
        cv2.normalize(img_rgb, img_rgb, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0]

    def _YCbCr_to_RGB(self):
        pass

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
