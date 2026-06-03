import torch
from typing import Any, Optional, Union

import cv2
import numpy as np


def canny_processor(frame: np.ndarray, threshold_low: float,
                    threshold_high: float) -> np.ndarray:
    return cv2.Canny(frame, threshold_low, threshold_high)


def sobel_processor(frame: np.ndarray) -> np.ndarray:
    kwargs = dict(ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_x = cv2.Sobel(frame, cv2.CV_16S, 1, 0, **kwargs)
    grad_y = cv2.Sobel(frame, cv2.CV_16S, 0, 1, **kwargs)
    xy = np.stack([grad_x, grad_y])
    grad = np.linalg.norm(xy, axis=0)
    return grad


def _cuda_is_available() -> bool:
    return bool(
        hasattr(cv2, "cuda")
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
    )


def _gaussian_blur(frame: np.ndarray, sigma: float,
                   use_cuda: bool = False) -> np.ndarray:
    if sigma <= 0:
        return frame

    if use_cuda and _cuda_is_available():
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        gpu_blurred = cv2.cuda.createGaussianFilter(
            gpu_frame.type(),
            gpu_frame.type(),
            (0, 0),
            sigma,
            sigma,
        ).apply(gpu_frame)
        return gpu_blurred.download()

    return cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)


def difference_of_gaussians_processor(frame: np.ndarray,
                                      sigma_low: float = 1.0,
                                      sigma_high: float = 2.0,
                                      use_cuda: bool = False
                                      ) -> np.ndarray:
    if sigma_low <= 0 or sigma_high <= 0:
        raise ValueError("Gaussian sigmas must be strictly positive.")
    if sigma_low == sigma_high:
        raise ValueError("Difference of Gaussians requires distinct sigmas.")

    frame_f32 = frame.astype(np.float32, copy=False)
    blur_low = _gaussian_blur(frame_f32, sigma_low, use_cuda=use_cuda)
    blur_high = _gaussian_blur(frame_f32, sigma_high, use_cuda=use_cuda)
    return np.abs(blur_low - blur_high)


def image_preprocessing(frame: np.ndarray, method: str = "none",
                        **kwargs: Any) -> np.ndarray:
    method = method.lower()

    if method == "none":
        return frame
    if method == "canny":
        threshold_low = kwargs.get("threshold_low", 50)
        threshold_high = kwargs.get("threshold_high", 150)
        return canny_processor(frame, threshold_low=threshold_low,
                               threshold_high=threshold_high)
    if method == "sobel":
        processed = sobel_processor(frame)
        processed = scale_image(processed, f=255.0, use_max=True)
        return processed.astype(np.uint8)
    if method in {"dog", "difference_of_gaussians"}:
        sigma_low = kwargs.get("sigma_low", 1.0)
        sigma_high = kwargs.get("sigma_high", 2.0)
        use_cuda = kwargs.get("use_cuda", False)
        processed = difference_of_gaussians_processor(
            frame,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            use_cuda=use_cuda,
        )
        processed = scale_image(processed, f=255.0, use_max=True)
        return processed.astype(np.uint8)

    valid_methods = ["none", "canny", "sobel", "dog"]
    raise ValueError(f"Unknown preprocessing method '{method}'. "
                     f"Expected one of {valid_methods}.")


def to_n_dim(image: Union[np.ndarray, torch.Tensor], n: Optional[int] = 3
             ) -> Union[np.ndarray, torch.Tensor]:
    while image.ndim < n:
        if isinstance(image, torch.Tensor):
            image = torch.unsqueeze(image, 0)
        else:
            image = np.expand_dims(image, 0)
    return image


def scale_image(image: Union[np.ndarray, torch.Tensor],
                f: Optional[float] = None, use_max: Optional[bool] = False
                ) -> Union[np.ndarray, torch.Tensor]:
    if use_max:
        m = np.max if isinstance(image, np.ndarray) else torch.max
        max_val = m(image)
        if max_val > 0:
            image = image / max_val
    if f is not None:
        image = image * f
    return image
