import math
import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from .FDA import FDA_source_to_target
from PIL import Image

def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1 # [256, 256, 2]
    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1) # [cos, sin, 2]
    tile_grads = lambda slice1, slice2: cv2.resize(np.repeat(np.repeat(gradients[slice1[0]: slice1[1], slice2[0]: slice2[1]], d[0], axis=0), d[1], axis=1), dsize=(shape[1], shape[0]))
    dot = lambda grad, shift: (np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]), axis=-1) * grad[: shape[0], : shape[1]]).sum(axis=-1)
    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0]) # [256, 256]
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0]) # [256, 256]
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1]) # [256, 256]
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1]) # [256, 256]
    t = fade(grid[:shape[0], :shape[1]]) # [256, 256, 2]
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])

rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

def fourier_perlin_noise(image, dtd_image, aug_prob=1.0):
    image = np.array(image, dtype=np.float32) # [256, 256, 3]
    dtd_image = np.array(dtd_image, dtype=np.float32) # [256, 256, 3]
    shape = image.shape[:2] # [256, 256]
    min_perlin_scale, max_perlin_scale = 0, 6
    t_x = torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0]
    t_y = torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0]
    perlin_scalex, perlin_scaley = 2**t_x, 2**t_y
    perlin_noise = rand_perlin_2d_np(shape, (perlin_scalex, perlin_scaley)) # [256, 256]
    perlin_noise = rot(images=perlin_noise) # [256, 256]
    perlin_noise = np.expand_dims(perlin_noise, axis=2) # [256, 256, 1]
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise)) # convert matrix to [0, 1] by threshold
    img_copy = image.copy().transpose((2, 0, 1))
    dtd_copy = dtd_image.copy().transpose((2, 0, 1))
    FDA_img = FDA_source_to_target(img_copy, dtd_copy).transpose((1, 2, 0))
    FDA_img = FDA_img / 255.0
    beta = torch.rand(1).numpy()[0] * 0.8
    image_aug = (1 - perlin_thr) * FDA_img + (1 - beta) * FDA_img + (beta * perlin_thr) * FDA_img # (beta + perlin_thr) * FDA_img
    image_aug = image_aug.astype(np.float32)
    return image_aug

if __name__ == '__main__':
    image = Image.open('000.png').convert('RGB').resize((256, 256), Image.BILINEAR)
    dtd_image = Image.open('frilly_0016.jpg').convert('RGB').resize((256, 256), Image.BILINEAR)
    aug = fourier_perlin_noise(image, dtd_image)
    cv2.imwrite('aug.jpg', (aug * 255.0)[:, :, ::-1])