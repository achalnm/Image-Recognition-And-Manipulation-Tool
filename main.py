from flask import Flask, render_template, request, send_file
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import torch
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    labels = outputs[0]['labels'].numpy()
    scores = outputs[0]['scores'].detach().numpy()
    boxes = outputs[0]['boxes'].detach().numpy()
    return labels, scores, boxes

def apply_effect(image, effect_type, intensity):
    intensity = max(0, min(intensity, 10))
    factor = intensity / 10.0

    if effect_type == 'BLUR':
        radius = factor * 20
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    elif effect_type == 'CONTOUR':
        image = image.filter(ImageFilter.CONTOUR)
        if factor > 0:
            image = ImageOps.autocontrast(image)
    
    elif effect_type == 'SHARPEN':
        for _ in range(int(factor * 2)):
            image = image.filter(ImageFilter.SHARPEN)
    
    elif effect_type == 'DETAIL':
        image = image.filter(ImageFilter.DETAIL)
        if factor > 0:
            image = ImageOps.autocontrast(image, cutoff=10 + int(factor * 30))
    
    elif effect_type == 'EMBOSS':
        image = image.filter(ImageFilter.EMBOSS)
        if factor > 0:
            image = ImageOps.autocontrast(image, cutoff=10 + int(factor * 20))
    
    return image

def enhance_image(image_bytes, effect_type, intensity):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = apply_effect(image, effect_type, intensity)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if file:
        image_bytes = file.read()
        labels, scores, boxes = get_prediction(image_bytes)
        effect_type = request.form['effect']
        intensity = float(request.form['intensity'])
        enhanced_image = enhance_image(image_bytes, effect_type, intensity)
        img_io = io.BytesIO()
        enhanced_image.save(img_io, 'JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
