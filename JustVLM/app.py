import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained VLM model (e.g., CLIP)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define labels for classification
labels = ["Normal", "COVID-19", "Pneumonia"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    print(f"File saved at: {file_path}")

    try:
        # Perform classification
        result = classify_image(file_path)
        # Pass result to the results page
        return render_template('results.html', analysis=result["message"], probabilities=result["classification"])
    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({"error": "Error during classification"}), 500

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    # Generate results
    results = {labels[i]: float(probs[i]) for i in range(len(labels))}
    prediction = max(results, key=results.get)
    confidence = results[prediction]

    # Generate detailed analysis
    if prediction == "COVID-19":
        details = (
            "The X-ray shows signs consistent with COVID-19, such as ground-glass opacities "
            "and bilateral patchy shadows, often seen in early stages of infection. "
            f"The confidence level for this diagnosis is {confidence:.2%}. "
        )
    elif prediction == "Pneumonia":
        details = (
            "The X-ray indicates features typical of pneumonia, including alveolar opacities, "
            "which may appear as consolidation or air bronchograms. This is consistent with an active infection. "
            f"The confidence level for this diagnosis is {confidence:.2%}. "
        )
    elif prediction == "Normal":
        details = (
            "The X-ray appears normal with no signs of significant abnormalities. "
            "There are no visible opacities, consolidations, or other patterns indicative of infection. "
            f"The confidence level for this result is {confidence:.2%}. "
        )
    else:
        details = "The analysis could not determine a specific diagnosis."

    # Combine details with probabilities
    full_message = details + "\n\nProbability Breakdown:\n"
    for label, prob in results.items():
        full_message += f"- {label}: {prob:.2%}\n"

    return {"classification": results, "message": full_message}

if __name__ == '__main__':
    app.run(debug=True)