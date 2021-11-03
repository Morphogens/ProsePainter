import base64
import io

from PIL import Image


def decode_base64_img(base64_img, ):
    base64_img = base64_img.split(",")[-1]
    img = base64.b64decode(base64_img)
    img = io.BytesIO(img)
    img = Image.open(img)

    return img


def pil_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, "jpeg")

    encoded = base64.standard_b64encode(buffer.getvalue()).decode()

    return encoded
