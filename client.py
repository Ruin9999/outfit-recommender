import requests
import base64
from PIL import Image
from io import BytesIO
from utils import save_images, get_timestamp

VERSION = 4

response = requests.post("http://127.0.0.0:8000/predict", json={"input": {
    "prompt" : "A man getting ready for an interview, he is wearing a suit and tie.",
    "controlnet_image_url": "https://avid-tapir-423.convex.cloud/api/storage/c765650a-0117-42c1-9d7d-ceac5fcd5a71"
}})

response = response.json()
image = response.get("image")

image_bytes = base64.b64decode(image)
image = Image.open(BytesIO(image_bytes))

timestamp = get_timestamp()
save_images([image], output_dir=f"outputs/version_{VERSION}", filename=timestamp)