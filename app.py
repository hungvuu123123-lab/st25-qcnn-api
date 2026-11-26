import io, os, base64, cv2, torch, joblib
import numpy as np
import torch.nn as nn
import pennylane as qml
from PIL import Image
import pillow_heif
from fastapi import FastAPI, File, UploadFile

pillow_heif.register_heif_opener()

device = torch.device("cpu")
scaler = joblib.load("feature_scaler.pkl")

n_qubits, q_layers = 4, 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[:, i] * np.pi, wires=i)
    for l in range(q_layers):
        for i in range(n_qubits):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 3, 16), nn.ReLU(), nn.Linear(16, n_qubits), nn.Tanh()
        )
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, {"weights": (q_layers, n_qubits, 2)})
        self.classifier = nn.Linear(n_qubits, 1)

    def forward(self, img, morph):
        img_feat = self.cnn(img)
        q_in = self.fusion(torch.cat([img_feat, morph], dim=1))
        q_out = self.qlayer(q_in)
        return self.classifier(q_out).squeeze(1)

model = HybridQCNN().to(device)
model.load_state_dict(torch.load("best_hybrid_qcnn.pth", map_location=device))
model.eval()

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ST25 Hybrid QCNN API is running on Render"}

@app.post("/analyze")
async def analyze_rice(file: UploadFile = File(...)):
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    target_w = 1200
    img = cv2.resize(img_bgr, (target_w, int(h * (target_w / float(w)))), interpolation=cv2.INTER_AREA)

    if np.mean(img[:, :int(target_w * 0.15)]) > 110:
        img = img[:, int(target_w * 0.22):]

    annotated_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    grain_imgs, grain_morphs, grain_boxes, aspect_ratios = [], [], [], []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 450 or area > 7500:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if x <= 4 or y <= 4 or (x + bw) >= (img.shape[1] - 4) or (y + bh) >= (img.shape[0] - 4):
            continue
        rect = cv2.minAreaRect(cnt)
        major_axis, minor_axis = max(rect[1]), min(rect[1])
        if minor_axis == 0:
            continue
        ar = major_axis / minor_axis
        if ar < 1.8 or ar > 6.5:
            continue

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_int = cv2.mean(gray, mask=mask)[0]

        pad = 6
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + bw + pad), min(img.shape[0], y + bh + pad)
        crop = img[y1:y2, x1:x2]
        gh, gw, _ = crop.shape
        ms = max(gh, gw)
        sq = np.zeros((ms, ms, 3), dtype=np.uint8)
        sq[(ms - gh)//2:(ms - gh)//2 + gh, (ms - gw)//2:(ms - gw)//2 + gw] = crop
        res = cv2.resize(sq, (64, 64), interpolation=cv2.INTER_AREA)

        grain_imgs.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        grain_morphs.append([ar, mean_int, area])
        grain_boxes.append((x, y, bw, bh))
        aspect_ratios.append(ar)

    total_grains = len(grain_imgs)
    if total_grains == 0:
        return {"error": "Không phát hiện được hạt gạo nào. Vui lòng chụp trên nền đen rõ nét!"}

    img_tensor = torch.tensor(np.array(grain_imgs), dtype=torch.float32).permute(0, 3, 1, 2).to(device) / 255.0
    morph_scaled = scaler.transform(np.array(grain_morphs))
    morph_tensor = torch.tensor(morph_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(img_tensor, morph_tensor)).cpu().numpy()

    st25_count = 0
    confidences = []
    for i, prob in enumerate(probs):
        bx, by, bw, bh = grain_boxes[i]
        if prob >= 0.5:
            st25_count += 1
            confidences.append(prob)
            cv2.rectangle(annotated_img, (bx, by), (bx + bw, by + bh), (46, 204, 113), 2)
        else:
            confidences.append(1.0 - prob)
            cv2.rectangle(annotated_img, (bx, by), (bx + bw, by + bh), (52, 73, 235), 2)

    purity_pct = round((st25_count / total_grains) * 100, 1)
    avg_conf = round(float(np.mean(confidences)) * 100, 1)
    avg_ar = round(float(np.mean(aspect_ratios)), 2)

    _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "purity_percent": purity_pct,
        "total_grains": total_grains,
        "st25_count": st25_count,
        "confidence_percent": avg_conf,
        "aspect_ratio": avg_ar,
        "annotated_image_base64": img_base64
    }