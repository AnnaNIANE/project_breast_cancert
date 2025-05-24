from flask import Flask, request, render_template
import torch
import tensorflow as tf
from PIL import Image
import numpy as np
from torchvision import transforms
from models.cnn_torch import CNN_Torch
from models.cnn_tensorflow import create_cnn_tensorflow
from utils.preprocessing import get_data
import os

app = Flask(__name__)

# Load class names
try:
    _, _, class_names = get_data()
    print(f"Loaded classes: {class_names}")
except Exception as e:
    print(f"Error loading dataset classes: {e}")
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load models
device = torch.device("cpu")  # PythonAnywhere only supports CPU
print(f"Using device: {device}")

try:
    pytorch_model = CNN_Torch(num_classes=4).to(device)
    pytorch_model.load_state_dict(torch.load("anna_model.torch", map_location=device))
    pytorch_model.eval()
    print("PyTorch model loaded successfully")
except Exception as e:
    print(f"Error loading PyTorch model: {e}")
    pytorch_model = None

try:
    tensorflow_model = tf.keras.models.load_model("anna_model.tensorflow")
    print("TensorFlow model loaded successfully")
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")
    tensorflow_model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        print("Received POST request")
        print(f"Model: {request.form.get('model')}")
        print(f"Image uploaded: {bool(request.files.get('image'))}")
        try:
            model_choice = request.form['model']
            file = request.files['image']
            if file:
                print("Processing image")
                image = Image.open(file).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                print(f"Image tensor shape: {image_tensor.shape}")
                if model_choice == 'pytorch':
                    if pytorch_model is None:
                        error = "PyTorch model not loaded"
                    else:
                        with torch.no_grad():
                            image_tensor = image_tensor.to(device)
                            output = pytorch_model(image_tensor)
                            pred = output.argmax(dim=1).item()
                            prediction = class_names[pred]
                            print(f"PyTorch prediction: {prediction}")
                else:  # tensorflow
                    if tensorflow_model is None:
                        error = "TensorFlow model not loaded"
                    else:
                        image_np = image_tensor.numpy()
                        image_np = np.transpose(image_np, (0, 2, 3, 1))
                        print(f"TensorFlow input shape: {image_np.shape}")
                        pred = np.argmax(tensorflow_model.predict(image_np, verbose=0), axis=1)[0]
                        prediction = class_names[pred]
                        print(f"TensorFlow prediction: {prediction}")
            else:
                error = "No image uploaded"
        except Exception as e:
            error = f"Error processing image: {str(e)}"
            print(f"Error: {error}")
    print("Rendering index.html")
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port)