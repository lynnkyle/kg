import pywt
import numpy as np
import torch


def dwt_denoise_batch(embeddings, wavelet='db4', level=1, threshold=0.1):
    """
    :param embeddings: shape [B, D]的Tensor
    :param wavelet: 小波基
    :param level: 小波分解层数
    :param threshold: 阈值，用于soft-threshold去噪
    :return: 去噪后的embedding, shape [B, D]
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    denoised = []

    for emb in embeddings_np:
        coeffs = pywt.wavedec(emb, wavelet=wavelet, level=level)
        # 软阈值处理
        coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        # 重建
        rec = pywt.waverec(coeffs_denoised, wavelet=wavelet)
        # 保持维度一致
        rec = rec[:emb.shape[0]]
        denoised.append(rec)

    denoised_up = np.stack(denoised, axis=0)
    return torch.tensor(denoised_up, dtype=embeddings.dtype, device=embeddings.device)


if __name__ == '__main__':
    x = torch.randn(size=(32, 64))
    print(x)
    y = dwt_denoise_batch(x)
    print(y)
