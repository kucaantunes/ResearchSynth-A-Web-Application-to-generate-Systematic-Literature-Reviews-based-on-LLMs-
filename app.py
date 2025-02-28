import os
import torch
from flask import Flask, render_template, request
from PIL import Image
import clip
from transformers import pipeline
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set up image upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device)

# Define a custom model with modified last layers for classification (3 classes)
class CustomCLIPModel(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.fc1 = torch.nn.Linear(512, 512)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 3)

    def forward(self, x):
        features = self.clip_model.encode_image(x)
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = CustomCLIPModel(clip_model).to(device)
model.eval()

# LLMs for report generation
gpt2 = pipeline('text-generation', model='gpt2')
gptneo = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
bart = pipeline('summarization', model='facebook/bart-large-cnn')

def get_disease_details(disease):
    if disease == "Pneumonia":
        return """Pneumonia is a lung infection caused by bacteria, viruses, or fungi. 
        Symptoms include fever, cough, difficulty breathing, and chest pain. 
        Pneumonia can be caused by various pathogens, including Streptococcus pneumoniae, 
        Haemophilus influenzae, and viruses like the flu or COVID-19. 
        Treatment typically includes antibiotics for bacterial pneumonia, antivirals for viral pneumonia, 
        and supportive care like oxygen therapy and fluids. Treatment should be guided by clinical symptoms, 
        imaging results, and culture or PCR tests."""
    elif disease == "COVID-19":
        return """COVID-19 is a disease caused by the SARS-CoV-2 virus, affecting the respiratory system.
        Symptoms include fever, dry cough, fatigue, shortness of breath, and loss of taste or smell. 
        Some individuals may experience severe symptoms, leading to pneumonia, acute respiratory distress syndrome (ARDS), 
        and multi-organ failure. COVID-19 is primarily transmitted via respiratory droplets, 
        and prevention includes wearing masks, social distancing, and vaccination. Management may involve hospitalization, 
        oxygen therapy, and antivirals, with treatment tailored to the severity of the symptoms."""
    else:
        return "No disease detected. The X-ray shows normal, and further diagnostic processes are recommended."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', message="No file uploaded.")
    
    file = request.files['image']
    image_filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    file.save(image_path)

    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_input)
        _, predicted_class = output.max(1)

    condition = ""
    if predicted_class == 0:
        condition = "Normal"
    elif predicted_class == 1:
        condition = "COVID-19"
    else:
        condition = "Pneumonia"

    llm_choice = request.form.get("llm", "gpt2")

    # Generate a single detailed report (selecting GPT-2 as the default)
    if llm_choice == "gpt2":
        report = gpt2(f"Disease detected: {condition}", max_length=250)[0]['generated_text']
    elif llm_choice == "gptneo":
        report = gptneo(f"Disease detected: {condition}", max_length=250)[0]['generated_text']
    else:
        report = bart(f"Disease detected: {condition}", max_length=250)[0]['summary_text']

    disease_details = get_disease_details(condition)

    return render_template('result.html', condition=condition, report=report, disease_details=disease_details, image_filename=image_filename)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get("user_input", "")
    llm_choice = request.form.get("llm", "gpt2")

    if llm_choice == "gpt2":
        response = gpt2(f"Medical conversation: {user_input}", max_length=250)[0]['generated_text']
    elif llm_choice == "gptneo":
        response = gptneo(f"Medical conversation: {user_input}", max_length=250)[0]['generated_text']
    else:
        response = bart(f"Medical conversation: {user_input}", max_length=250)[0]['summary_text']

    return response

if __name__ == '__main__':
    app.run(debug=True)
