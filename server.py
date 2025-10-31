import torch
import torch.nn.functional as F
from Adafruit_IO import MQTTClient
from PIL import Image
import base64
import io
import numpy as np
from torchvision import models

AIO_USERNAME = "teja1845"
AIO_KEY = "aio_EWkz38xDp8dhG3Q1qz7V7YbGr0Dm"
FEED_IMAGE = "camera-image"
FEED_RESULT = "rodent-result"
MODEL_PATH = "rodent_detector_finetuned.pt"
IMG_SIZE = 224
DEVICE = torch.device("cpu")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model_state_dict = checkpoint['model_state_dict']

model = models.mobilenet_v2(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(num_features, 512),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(512),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(256),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, 2)
)
model.load_state_dict(model_state_dict)
model.eval()
print("[INFO] Model loaded successfully and ready for inference.")

def preprocess_image(pil_img):
    img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    img = (img - mean) / std
    tensor = torch.tensor(img).unsqueeze(0)
    return tensor

def classify_image(image_b64):
    try:
        image_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess_image(pil_img)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        result = "rodent" if pred_class == 1 else "no_rodent"
        print(f"[RESULT] {result.upper()} ({confidence*100:.2f}% confidence)")
        return result
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        return "error"

def connected(client):
    print("[INFO] Connected to Adafruit IO")
    client.subscribe(FEED_IMAGE)
    print(f"[INFO] Subscribed to feed '{FEED_IMAGE}'")

def disconnected(client):
    print("[WARN] Disconnected from Adafruit IO")
    exit(1)

def message(client, feed_id, payload):
    print(f"[MQTT] New image received on '{feed_id}'")
    result = classify_image(payload)
    client.publish(FEED_RESULT, result)
    print(f"[MQTT] Published classification result: {result}")

if __name__ == "__main__":
    client = MQTTClient(AIO_USERNAME, AIO_KEY)
    client.on_connect = connected
    client.on_disconnect = disconnected
    client.on_message = message
    print("[INFO] Connecting to Adafruit IO...")
    client.connect()
    client.loop_blocking()
