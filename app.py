from flask import Flask, render_template, request
import os
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel


app = Flask(__name__)


UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    description = request.form.get('description', '').strip()

    if file.filename == '':
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "File type not allowed. Please upload PNG or JPG.", 400

    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

  
    image = Image.open(filepath).convert('RGB')
    inputs = processor(text=[description], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

 
    image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)

    similarity = torch.cosine_similarity(image_embeds, text_embeds).item()

 
    if similarity > 0.30:  
        result = " Correct – Image matches description"
    else:
        result = " Incorrect – Image does not match description"

    return render_template('result.html', image=filepath, desc=description, result=result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True) 