from os.path import join
from pathlib import Path
import copy

import numpy as np
import torch
from torch import nn
import cv2


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def robust_min(img, p=5):
    return np.percentile(img.ravel(), p)


def robust_max(img, p=95):
    return np.percentile(img.ravel(), p)


def normalize(img, q_min=10, q_max=90):
    """
    robust min/max normalization if specified with norm arguments
    q_min and q_max are the min and max quantiles for the normalization
    :param img: Input image to be normalized
    :param q_min: min quantile for the robust normalization
    :param q_max: max quantile for the robust normalization
    :return: Normalized image
    """
    norm_min = robust_min(img, q_min)
    norm_max = robust_max(img, q_max)
    normalized = (img - norm_min) / (norm_max - norm_min)
    return normalized


def calculate_histogram(img, bins=4):
    """
    Calculate the histogram of the input image
    :param img: Input image
    :param bins: Number of bins for the histogram
    :return: Histogram of the input image
    """
    hist = cv2.calcHist([img*256], [0], None, [bins], [0, 256])
    return hist.astype(np.int32)


def torch2cv2(image):
    image = torch.squeeze(image)  # H x W
    image = image.cpu().numpy()
    return image


def cv2torch(image, num_ch=1):
    img_tensor = torch.tensor(image)
    if len(img_tensor.shape) == 2:
        img_tensor = torch.unsqueeze(img_tensor, 0)
        if num_ch > 1:
            img_tensor = img_tensor.repeat(num_ch, 1, 1)
    if len(img_tensor.shape) == 3:
        img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor


def append_timestamp(path, description, timestamp):
    with open(path, 'a') as f:
        f.write('{} {:.15f}\n'.format(description, timestamp))


def append_result(path, description, result):
    with open(path, 'a') as f:
        if isinstance(result, list):
            for idx, elem in zip(description, result):
                f.write('{} {:.5f}\n'.format(idx, elem))
        else:
            f.write('{} {:.5f}\n'.format(description, result))


def append_result_int(path, description, result):
    with open(path, 'a') as f:
        f.write('{} {}\n'.format(description, result))


def setup_output_folder(output_folder):
    """
    Ensure existence of output_folder and overwrite output_folder/timestamps.txt file.
    Returns path to output_folder/timestamps.txt
    """
    ensure_dir(output_folder)
    print('Saving to: {}'.format(output_folder))


def save_inferred_image(folder, image, idx, tiff=False):
    png_name = 'frame_{:010d}.png'.format(idx)
    png_path = join(folder, png_name)
    image_for_png = np.round(image * 255).astype(np.uint8)
    cv2.imwrite(png_path, image_for_png)
    if tiff:
        tiff_name = 'frame_{:010d}.tiff'.format(idx)
        tiff_path = join(folder, tiff_name)
        image_for_tiff = image
        cv2.imwrite(tiff_path, image_for_tiff)


def print_epoch(checkpoint):
    try:
        epoch = checkpoint['epoch']
        print("Using model weights after epoch {}".format(epoch))
    except KeyError:
        print("Unable to get epoch number, might be legacy E2VID checkpoint")


def get_epoch(checkpoint):
    if 'epoch' in checkpoint:
        return checkpoint['epoch']
    else:
        return -1


def get_arch(checkpoint):
    if 'arch' in checkpoint:
        return checkpoint['arch']
    else:
        return ""


def profile_model(model, num_bins, size):
    from thop import profile
    from inspect import signature
    num_inputs = len(signature(model.forward).parameters)
    if num_inputs == 1:
        inputs = (torch.randn(1, num_bins, size, size, device="cuda"),)
    elif num_inputs == 2:
        inputs = (torch.randn(1, num_bins, size, size, device="cuda"),
                  torch.randn(1, 1, device="cuda"))
    elif num_inputs == 3:
        inputs = (torch.randn(1, num_bins, size, size, device="cuda"),
                  torch.randn(1, 1, device="cuda"),
                  torch.randn(1, 1, device="cuda"))
    else:
        raise ValueError("Unsupported number of inputs: {}".format(num_inputs))
    if isinstance(model, torch.nn.DataParallel):
        thop_model = model.module
    else:
        thop_model = model
    macs, params = profile(copy.deepcopy(thop_model), inputs=inputs)
    return macs, params


def set_bn_training(m):
    if isinstance(m, nn.BatchNorm2d):
        m.train()


def set_bn_track_running_stats_false(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False
