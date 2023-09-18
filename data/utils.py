import torchstain

from torchvision import transforms
from skimage.exposure import match_histograms
from skimage import io


import numpy as np


class ColourNormalization():
    def __init__(self,  target_image_path):

        target_image = np.array(io.imread(target_image_path))
        torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])
        torch_normalizer.fit(T(target_image))

        self.transform=T
        self.normalizer=torch_normalizer
        self.target_image = target_image

    def __call__(self, image, *args, **kwargs):

        image = match_histograms(image, self.target_image, channel_axis=-1)
        image = self.transform(image)
        norm, _, _ = self.normalizer.normalize(I=image, stains=False)
        norm = norm.numpy()
        return norm