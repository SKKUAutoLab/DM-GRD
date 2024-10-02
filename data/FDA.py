import numpy as np
from PIL import Image
import cv2

def low_freq_mutate_np(amp_src, amp_dest, L=0.1):
    lam = np.random.uniform(0.0, 1.0)
    # shift zero-frequency of src and dest to center spectrum
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_dest = np.fft.fftshift(amp_dest, axes=(-2, -1))
    # mix-up in frequency domain
    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1
    a_src_ = np.copy(a_src)
    a_dest_ = np.copy(a_dest)
    a_src[:, h1:h2, w1:w2] = lam * a_dest_[:, h1:h2, w1:w2] + (1 - lam) * a_src_[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src

def FDA_source_to_target(src_img, dest_img, L=0.1):
    # get fft of both source and target
    fft_src = np.fft.fft2(src_img, axes=(-2, -1))
    fft_dest = np.fft.fft2(dest_img, axes=(-2, -1))
    # extract amplitude and phase of both source and target
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    amplitude_dest, phase_dest = np.abs(fft_dest), np.angle(fft_dest)
    # mix-up low frequency of src and dest
    amp_src_ = low_freq_mutate_np(amplitude_src, amplitude_dest, L=L)
    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * phase_src)
    # get the mutated image
    src_in_dest = np.real(np.fft.ifft2(fft_src_, axes=(-2, -1)))
    # src_in_dest = np.uint8(np.clip(src_in_dest, 0, 255))
    return src_in_dest

if __name__ == '__main__':
    src_img = Image.open('000.png').convert('RGB').resize((256, 256), Image.BICUBIC)
    src_img = np.asarray(src_img, np.float32)[:, :, ::-1]
    src_img = src_img.transpose((2, 0, 1))
    target_img = Image.open('banded_0002.jpg').convert('RGB').resize((256, 256), Image.BICUBIC)
    target_img = np.asarray(target_img, np.float32)[:, :, ::-1]
    target_img = target_img.transpose((2, 0, 1))
    out = FDA_source_to_target(src_img, target_img, 0.1)
    out = out.transpose((1, 2, 0))
    cv2.imwrite('aug.jpg', out)