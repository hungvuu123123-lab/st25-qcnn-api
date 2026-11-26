import io, os, gc, base64, cv2, torch, joblib
import numpy as np
import torch.nn as nn
import pennylane as qml
from PIL import Image
import pillow_heif
from fastapi import FastAPI, File, UploadFile

pillow_heif.register_heif_opener()

device = torch.device("cpu")
scaler = joblib.load("feature_scaler.pkl")

n_qubits, q_layers = 6, 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Cấu trúc Data Re-uploading khớp 100% với bản huấn luyện 6-Qubit trên Colab
    for l in range(q_layers):
        for i in range(n_qubits):
            qml.RY(inputs[:, i] * np.pi, wires=i)
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
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 6, 32), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(32, n_qubits), nn.Tanh()
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
    return {"status": "ST25 Hybrid QCNN (6-Qubit VQC) API is running on Render"}

@app.post("/analyze")
async def analyze_rice(file: UploadFile = File(...)):
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    target_w = 1200
    img = cv2.resize(img_bgr, (target_w, int(h * (target_w / float(w)))), interpolation=cv2.INTER_AREA)

    left_strip = cv2.cvtColor(img[:, :int(target_w * 0.16)], cv2.COLOR_BGR2GRAY)
    if np.mean(left_strip > 130) > 0.65:
        img = img[:, int(target_w * 0.23):]

    annotated_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_grains, areas = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400 or area > 35000:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > img.shape[1] * 0.6 or bh > img.shape[0] * 0.6:
            continue
        if x <= 2 or y <= 2 or (x + bw) >= (img.shape[1] - 2) or (y + bh) >= (img.shape[0] - 2):
            continue
        rect = cv2.minAreaRect(cnt)
        major_axis, minor_axis = max(rect[1]), min(rect[1])
        if minor_axis == 0:
            continue
        ar = major_axis / minor_axis
        if ar < 1.8 or ar > 6.5:
            continue

        valid_grains.append((cnt, area, x, y, bw, bh, ar))
        areas.append(area)

    total_grains = len(valid_grains)
    if total_grains == 0:
        return {"error": "Không phát hiện được hạt gạo nào. Vui lòng chụp trên nền đen rõ nét!"}

    median_area = float(np.median(areas))
    grain_imgs, grain_morphs, grain_boxes, aspect_ratios = [], [], [], []

    for cnt, area, x, y, bw, bh, ar in valid_grains:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_int = cv2.mean(gray, mask=mask)[0]
        std_int = cv2.meanStdDev(gray, mask=mask)[1][0][0]
        rel_area = area / median_area

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0
        perim = cv2.arcLength(cnt, True)
        compactness = (perim ** 2) / area if area > 0 else 0.0

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
        grain_morphs.append([ar, mean_int, std_int, rel_area, solidity, compactness])
        grain_boxes.append((x, y, bw, bh))
        aspect_ratios.append(ar)

    img_tensor = torch.tensor(np.array(grain_imgs), dtype=torch.float32).permute(0, 3, 1, 2).to(device) / 255.0
    morph_scaled = scaler.transform(np.array(grain_morphs))
    morph_tensor = torch.tensor(morph_scaled, dtype=torch.float32).to(device)

    # Chia nhỏ lô 16 hạt để giữ mức RAM ~250MB, chống lỗi 502 khi bấm phân tích liên tục
    probs_list = []
    with torch.no_grad():
        for start in range(0, total_grains, 16):
            end = start + 16
            batch_probs = torch.sigmoid(model(img_tensor[start:end], morph_tensor[start:end])).cpu().numpy()
            probs_list.append(batch_probs)
    probs = np.concatenate(probs_list, axis=0)

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

    del img_tensor, morph_tensor, grain_imgs, pil_img, img_bgr, img, gray, blurred, thresh
    gc.collect()

    return {
        "purity_percent": purity_pct,
        "total_grains": total_grains,
        "st25_count": st25_count,
        "confidence_percent": avg_conf,
        "aspect_ratio": avg_ar,
        "annotated_image_base64": img_base64
    }
