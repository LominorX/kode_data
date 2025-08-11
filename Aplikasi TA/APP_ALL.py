from flask import Flask, request, render_template
import torch
import numpy as np
import os
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import platform
import psutil

app = Flask(__name__)

# === KONFIGURASI MODEL BEATS ===
MODEL_DIR_BEATS = r'D:\KULIAH\TELKOM_UNIVERSITY\SEMESTER_8\TA\TA_SKRIPSI_GUE\APLIKASI\App_Beats\B_fold2'
LABEL_MAP_BEATS = {0: 'N', 1: 'L', 2: 'R', 3: 'V', 4: 'Q'}
MAX_LEN_BEATS = 512

tokenizer_beats = BertTokenizer.from_pretrained(MODEL_DIR_BEATS)
model_beats = BertForSequenceClassification.from_pretrained(MODEL_DIR_BEATS)
model_beats.eval()

# === KONFIGURASI MODEL RYTHM ===
MODEL_PATH_RYTHM = r'D:\KULIAH\TELKOM_UNIVERSITY\SEMESTER_8\TA\TA_SKRIPSI_GUE\APLIKASI\App_Rhythm\Hasil_FINAL_ENCODER_RYTHM\encoder_best.pt'
LABEL_MAP_RYTHM = {0: 'AFIB', 1: 'VFL', 2: 'VT'}
MAX_LEN_RYTHM = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === DEFINISI MODEL RYTHM ===
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=MAX_LEN_RYTHM, d_model=768, nhead=12, num_layers=12, num_classes=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x.unsqueeze(1))
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

model_rythm = SimpleEncoder(num_classes=len(LABEL_MAP_RYTHM)).to(DEVICE)
state_dict = torch.load(MODEL_PATH_RYTHM, map_location=DEVICE)
model_rythm.load_state_dict(state_dict)
model_rythm.eval()

# === FUNGSI MONITOR SPESIFIKASI SISTEM ===
def get_system_info():
    info = {
        'CPU Model': platform.processor(),
        'Physical Cores': psutil.cpu_count(logical=False),
        'Logical Cores': psutil.cpu_count(logical=True),
        'Total RAM (GB)': round(psutil.virtual_memory().total / (1024 ** 3), 2),
        'Device': str(DEVICE),
    }
    if torch.cuda.is_available():
        info['GPU Model'] = torch.cuda.get_device_name(0)
        info['GPU Memory (GB)'] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    else:
        info['GPU Model'] = 'Not Available'
        info['GPU Memory (GB)'] = 0
    return info

# === FUNGSI PREDIKSI BEATS ===
def predict_beats(signal_array):
    norm = ((signal_array - signal_array.min()) / (signal_array.ptp() + 1e-8) * 255).astype(int)
    signal_text = " ".join(map(str, norm.tolist()))
    tokens = tokenizer_beats(signal_text, return_tensors='pt', max_length=MAX_LEN_BEATS, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model_beats(**tokens)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_label].item()
    return LABEL_MAP_BEATS[pred_label], confidence

# === FUNGSI PREDIKSI RYTHM ===
def preprocess_signal(sig, target_len=MAX_LEN_RYTHM):
    if len(sig) < target_len:
        pad = np.full(target_len - len(sig), sig[-1])
        sig = np.concatenate([sig, pad])
    else:
        idx = np.linspace(0, len(sig) - 1, target_len).astype(int)
        sig = sig[idx]
    sig = (sig - sig.min()) / (sig.ptp() + 1e-8)
    return torch.tensor(sig, dtype=torch.float32)

def predict_rythm(signal_array):
    tensor_signal = preprocess_signal(signal_array).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model_rythm(tensor_signal)
        probs = torch.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_label].item()
    return LABEL_MAP_RYTHM[pred_label], confidence

# === ROUTES ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_beats', methods=['POST'])
def predict_beats_route():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    signal = np.load(file)
    label, confidence = predict_beats(signal)
    system_info = get_system_info()
    return render_template('result.html', label=label, confidence=round(confidence*100, 2), model='Beats', system_info=system_info)

@app.route('/predict_rythm', methods=['POST'])
def predict_rythm_route():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    signal = np.load(file)
    label, confidence = predict_rythm(signal)
    system_info = get_system_info()
    return render_template('result.html', label=label, confidence=round(confidence*100, 2), model='Rhythm', system_info=system_info)

if __name__ == '__main__':
    app.run(debug=True)