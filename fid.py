
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy import linalg
from torch.autograd import Variable

class FIDScore:
    def __init__(self, device='cuda', batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        # Remove the classification layer, we only need features
        self.inception.fc = nn.Identity()

    def get_activations(self, images):
        """
        Calculates the activations of the pool_3 layer for all images.
        images: Torch tensor of shape (N, C, H, W)
        """
        # Resize to 299x299 as required by Inception v3
        upsample = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        
        pred_arr = np.empty((images.size(0), 2048))
        
        # Process in batches
        with torch.no_grad():
            batch = upsample(images)
            # Inception expects inputs in range [0, 1] (or normalized, depending on transform_input)
            # Standard PyTorch Inceptionv3 expects:
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Assuming images are [-1, 1], convert to [0, 1] then normalize
            # But here we might just pass them if they are somewhat reasonable or assume caller handles it.
            # Let's assume input is [-1, 1].
            batch = (batch + 1) / 2.0 # [0, 1]
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)
            batch = (batch - mean) / std

            pred = self.inception(batch)
            
            # If pred is a tuple (aux), take the first one
            if isinstance(pred, tuple):
                pred = pred[0]
                
            pred_arr = pred.cpu().numpy()

        return pred_arr

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print('fid calculation produces singular product; '
                  'adding %s to diagonal of cov estimates' % eps)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def calculate_fid(self, real_images, fake_images):
        """
        Calculates the FID for two sets of images.
        real_images: Tensor (N, C, H, W)
        fake_images: Tensor (N, C, H, W)
        """
        act1 = self.get_activations(real_images)
        act2 = self.get_activations(fake_images)

        mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

        fid_value = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value
