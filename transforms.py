import torch
import torchvision
import numpy as np

from typing import List


def invert_single_full_transformation_matrix(theta):
    a, b, c, d = theta[0, 0], theta[0, 1], theta[1, 0], theta[1, 1]
    det_sub = a * d - b * c
    inv_theta = np.zeros_like(theta)
    inv_theta[0, 0] = d
    inv_theta[0, 1] = -b
    inv_theta[1, 0] = -c
    inv_theta[1, 1] = a
    inv_theta = inv_theta / det_sub
    inv_theta[0, 2] = -theta[0, 2]
    inv_theta[1, 2] = -theta[1, 2]
    inv_theta[2, 2] = 1.
    return inv_theta


def crop_params_to_full_matrix(crop_params, height, width):
    theta = np.zeros((3, 3), dtype=np.float32)
    theta[2, 2] = 1.
    theta[0, 0] = crop_params[3] / width
    theta[1, 1] = crop_params[2] / height
    theta[0, 2] = theta[0, 0] - 1 + 2 * crop_params[1] / width
    theta[1, 2] = -theta[1, 1] + 1 - 2 * crop_params[0] / height
    return theta


class TransformsSimCLRReturnTransforms:
    def __init__(self, size, jitter_strength=.5, p_grayscale=.2, p_color_jitter=.8,
                 scale=(.08, 1.), ratio=(3 / 4, 4 / 3), random_flip=True):
        s = jitter_strength
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomApply([color_jitter], p=p_color_jitter),
                torchvision.transforms.RandomGrayscale(p=p_grayscale),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.random_flip = random_flip

    def __call__(self, x):
        width, height = torchvision.transforms.functional.get_image_size(x)
        crop_params_a = torchvision.transforms.RandomResizedCrop.get_params(x, self.scale, self.ratio)
        theta_a = crop_params_to_full_matrix(crop_params_a, height, width)
        crop_params_b = torchvision.transforms.RandomResizedCrop.get_params(x, self.scale, self.ratio)
        theta_b = crop_params_to_full_matrix(crop_params_b, height, width)

        a = torchvision.transforms.functional.resized_crop(x, *crop_params_a, self.size)
        if self.random_flip and torch.rand(1) < .5:
            a = torchvision.transforms.functional.hflip(a)
            theta_a[0, 0] = -theta_a[0, 0]

        b = torchvision.transforms.functional.resized_crop(x, *crop_params_b, self.size)
        if self.random_flip and torch.rand(1) < .5:
            b = torchvision.transforms.functional.hflip(b)
            theta_b[0, 0] = -theta_b[0, 0]

        ret_a = (theta_b @ invert_single_full_transformation_matrix(theta_a))[:2]  # from a to b
        ret_b = (theta_a @ invert_single_full_transformation_matrix(theta_b))[:2]  # from b to a

        return (self.transform(a), ret_a), (self.transform(b), ret_b)


def get_gaussian_kernel1d(kernel_size: int, sigma: torch.Tensor) -> torch.Tensor:
    ksize_half = (kernel_size - 1) * .5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).type_as(sigma)
    pdf = torch.exp(-.5 * (x[None, :] / sigma[:, None]).pow(2))
    kernel1d = pdf / pdf.sum(dim=1, keepdim=True)

    return kernel1d


def get_gaussian_kernel2d(kernel_size: List[int], sigma: torch.Tensor,
                          dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    kernel1d_x = get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = kernel1d_y[:, :, None] * kernel1d_x[:, None, :]
    return kernel2d


class RandomGaussian(torch.nn.Module):
    def __init__(self, size, p=.5, sigma_range=(.1, 2.)):
        super().__init__()
        kernel_size = int(0.1 * size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = (kernel_size, kernel_size)
        self.p = p
        self.sigma_range = sigma_range

    def forward(self, im):
        batch_size = im.size(0)
        sel = torch.less(torch.rand(batch_size).type_as(im), self.p)[:, None, None, None]

        sigma_tensor_1d = torch.empty(batch_size).uniform_(self.sigma_range[0], self.sigma_range[1]).type_as(im)
        sigma_tensor_2d = torch.stack((sigma_tensor_1d, sigma_tensor_1d), 0)
        gauss_kernel = get_gaussian_kernel2d(self.kernel_size, sigma_tensor_2d, im.dtype, im.device)[:, None]
        padding = [self.kernel_size[0] // 2, self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[1] // 2]
        padded_im = torch.nn.functional.pad(im, padding, mode='reflect')
        channel_first_img = torch.transpose(padded_im, 0, 1)
        channel_first_gauss_im = torch.nn.functional.conv2d(channel_first_img, gauss_kernel, groups=batch_size).type_as(im)
        gauss_im = torch.transpose(channel_first_gauss_im, 0, 1)

        im = torch.where(sel, gauss_im, im)
        return im


class RandomSolarize(torch.nn.Module):
    def __init__(self, p=.1):
        super().__init__()
        self.p = p

    def forward(self, im, return_selection=False):
        sel = torch.less(torch.rand(im.size(0)).type_as(im), self.p)
        full_sel = sel[:, None, None, None]
        solarized_im = torch.where(im < .5, im, 1. - im)
        im = torch.where(full_sel, solarized_im, im)
        if return_selection:
            return im, sel
        return im
