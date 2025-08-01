import pywt
import numpy as np
import torch


def wavelet_transform_embedding(embedding, wavelet='db1', level=3)
    if torch.is_tensor(embedding):
        embedding = embedding.cpu().numpy()
    coeffs = pywt.wavedec(embedding, wavelet, level=level)

    coeff_vec = np.concatenate(coeffs)
    coeff_tensor = torch.tensor(coeff_vec, dtype=torch.float32)
    coeff_tensor = coe
