import os
from typing import *

from loguru import logger
import torch
import torchvision
import requests
import numpy as np
from PIL import Image

from server.server_config import DEBUG, DEBUG_OUT_DIR, MAX_IMG_DIM, MIN_IMG_DIM


def process_mask(
    mask_pil: Image.Image,
    target_mask_size: Tuple = None,
) -> np.ndarray:
    """
    Convert PIL mask into a numpy array

    Args:
        mask_pil (Image.Image): pil mask
        target_mask_size (Tuple, optional): target size of the mask. Defaults to None.

    Returns:
        np.ndarray: processed mask.
    """
    if target_mask_size is not None:
        mask_pil = mask_pil.resize(target_mask_size)

    mask = np.float32(np.array(mask_pil)) / 255.
    mask = mask[:, :, -1]

    return mask


def get_limits_from_mask(
    mask: np.ndarray,
    padding_percent: int = 10,
) -> Tuple:
    """
    Use mask to extract vertical and horizontal limits where values are different than 0.

    Args:
        mask (np.ndarray): mask
        padding_percent (int, optional): percent of padding to add to the computed limits. Defaults to 10.

    Returns:
        Tuple: limits in the following order: min_h, max_h, min_w, max_w.
    """
    height, width = mask.shape
    w_pad = int(width * (padding_percent / 100))
    h_pad = int(height * (padding_percent / 100))

    w_accum = np.where(np.sum(
        mask,
        axis=0,
    ) > 0)[-1]
    w_limits = (
        max(0, w_accum[0] - w_pad),
        min(width, w_accum[-1] + w_pad),
    )

    h_accum = np.where(np.sum(
        mask,
        axis=1,
    ) > 0)[-1]
    h_limits = (
        max(0, h_accum[0] - h_pad),
        min(height, h_accum[-1] + h_pad),
    )

    return h_limits[0], h_limits[1], w_limits[0], w_limits[1]


def get_crop_tensor_from_img(
    img: np.ndarray,
    limits: Tuple,
) -> torch.Tensor:
    """
    Crop img using `limits` and convert to tensor.

    Args:
        img (np.ndarray): image to crop.
        limits (Tuple): limits in the following order: min_h, max_h, min_w, max_w.

    Returns:
        torch.Tensor: cropped img tensor.
    """
    img_crop = img[limits[0]:limits[1], limits[2]:limits[3], ]
    img_crop = torch.tensor(img_crop)[None, ...].permute(0, 3, 1, 2)

    return img_crop


def scale_crop_tensor(crop_tensor: torch.Tensor, ) -> torch.Tensor:
    """
    Use global config to scale `crop_tensor` if necessary.

    Args:
        crop_tensor (torch.Tensor): crop to scale.

    Returns:
        torch.Tensor: scaled crop.
    """
    crop_size = crop_tensor.shape[2::]
    logger.debug(f"IMAGE CROP SIZE: {crop_size}")

    if any([size > MAX_IMG_DIM for size in crop_size]):
        scale_factor = max(crop_size) / MAX_IMG_DIM
        scale_factor = scale_factor

        crop_size = tuple(np.asarray(crop_size) / scale_factor)

    elif all([size < MIN_IMG_DIM for size in crop_size]):
        scale_factor = max(crop_size) / MIN_IMG_DIM
        scale_factor = scale_factor

        crop_size = tuple(np.asarray(crop_size) / scale_factor)

    # NOTE: scale to the nearest multiples of 16
    crop_size = tuple(np.int32(np.ceil(crop_size) / 16) * 16)
    logger.debug(f"SCALED CROP SIZE: {crop_size}")

    crop_tensor = torch.nn.functional.interpolate(
        crop_tensor,
        crop_size,
        mode='bilinear',
        align_corners=True,
    )

    return crop_tensor


def merge_gen_img_into_canvas(
    gen_img: Union[torch.Tensor, np.ndarray, ],
    mask: Union[torch.Tensor, np.ndarray, ],
    canvas_img: np.ndarray,
    crop_limits: Tuple,
) -> torch.Tensor:
    """
    Merge generated image into the canvas.

    Args:
        gen_img (Union[torch.Tensor, np.ndarray, ]): generated image.
        mask (Union[torch.Tensor, np.ndarray, ]): mask.
        canvas_img (np.ndarray): canvas image.
        crop_limits (Tuple): limits in the following order: min_h, max_h, min_w, max_w.

    Returns:
        torch.Tensor: canvas with the generated image merged into it in the area represented in the mask.
    """
    if not torch.is_tensor(gen_img):
        gen_img = torch.tensor(gen_img[None, :]).permute(0, 3, 1, 2)

    if not torch.is_tensor(mask):
        mask = torch.tensor(mask[None, None, ...])

    gen_img = torch.nn.functional.interpolate(
        gen_img,
        (
            int((crop_limits[1] - crop_limits[0])),
            int((crop_limits[3] - crop_limits[2])),
        ),
        mode='bilinear',
        align_corners=True,
    )

    mask = torch.nn.functional.interpolate(
        mask,
        (
            int((crop_limits[1] - crop_limits[0])),
            int((crop_limits[3] - crop_limits[2])),
        ),
        mode='bilinear',
        align_corners=True,
    )

    if DEBUG:
        os.makedirs(
            DEBUG_OUT_DIR,
            exist_ok=True,
        )

        torchvision.transforms.ToPILImage(mode="RGB")(gen_img[0]).save(
            os.path.join(DEBUG_OUT_DIR, "final_gen.png"))
        torchvision.transforms.ToPILImage(mode="L")(mask[0]).save(
            os.path.join(DEBUG_OUT_DIR, "final_mask.png"))

    gen_img = gen_img[0].detach().cpu().permute(1, 2, 0).numpy()
    mask = mask[0].detach().cpu().permute(1, 2, 0).numpy()

    canvas_img[
        crop_limits[0]:crop_limits[1],
        crop_limits[2]:crop_limits[3], :] = canvas_img[
            crop_limits[0]:crop_limits[1],
            crop_limits[2]:crop_limits[3], :] * (1 - mask) + gen_img * mask

    return canvas_img


def _get_confirm_token(response, ):
    """"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination):
    """"""
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(
    google_drive_id: str,
    out_dir: str,
) -> None:
    """
    Download content from google drive with the provided google_drive_id in the selected out_dir.

    Args:
        google_drive_id (str): id of the file stored in google drive.
        out_dir (str): directory where the downloaded file will be stored.
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': google_drive_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': google_drive_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, out_dir)
