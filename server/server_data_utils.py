import base64
import io

from PIL import Image


def base64_to_pil(base64_img: str, ) -> Image.Image:
    """
    Convert a base64 image into a PIL Image.

    Args:
        base64_img (str): base 64 encoding of an image.

    Returns:
        Image.Image: decoded PIL image
    """
    base64_img = base64_img.split(",")[-1]
    img = base64.b64decode(base64_img)
    img = io.BytesIO(img)
    img = Image.open(img)

    return img


def pil_to_base64(
    img: Image.Image,
    img_format: str = "jpeg",
) -> str:
    """
    Encode a PIL image into base64.

    Args:
        img (Image.Image): PIL image
        img_format (str, optional): format of the PIL image. Defaults to "jpeg".

    Returns:
        str: image encoded in base64.
    """
    buffer = io.BytesIO()
    img.save(
        buffer,
        img_format,
    )

    base64_img = base64.standard_b64encode(buffer.getvalue()).decode()

    return base64_img
