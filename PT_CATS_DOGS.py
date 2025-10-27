import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import copy
import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# --- beállítások ---
DATA_ROOT = "/Users/kerekesdaniel/Projects/AI_Test/ai_framework_tests/Dataset_cats/Cat_vs_Dog"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "validation")
TEST_DIR = os.path.join(DATA_ROOT, "test")

IMG_SIZE = (300, 300)
BATCH_SIZE = 8
SEED = 123
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "best_model_pt.pth"

# --- seed ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# --- augmentációk (TF-hez hasonló) ---
# TF RandomRotation(0.06) ~ ±0.06*360° = ±21.6°
transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=21.6),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.94, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # [0,1]->[-1,1]
])

transform_val = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# --- datasets & dataloaders ---
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
val_ds = datasets.ImageFolder(VAL_DIR, transform=transform_val)
test_ds = datasets.ImageFolder(TEST_DIR, transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

class_names = train_ds.classes  # expected ["cat","dog"] 

# --- modell: MobileNetV2 ---
pt_base = models.mobilenet_v2(pretrained=True)
# Remove classifier, keep features + pooling
# torchvision's mobilenet_v2 has .features and .classifier; features output channels 1280
class MobileNetV2Binary(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.features  # feature extractor
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1280, 1)  # binary
    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # raw logits

model = MobileNetV2Binary(pt_base).to(DEVICE)

# --- segéd: freeze base ---
def set_base_trainable(model, trainable: bool, fine_tune_at: int = None):
    # If fine_tune_at is None: set all base params trainable flag to trainable
    # If fine_tune_at is given: freeze first fine_tune_at layers of base (by module order)
    for param in model.base.parameters():
        param.requires_grad = trainable
    if fine_tune_at is not None:
        # freeze first fine_tune_at layers in model.base (by children order)
        children = list(model.base.children())
        # iterate modules and count params layers; approximate by module index
        cum = 0
        for i, ch in enumerate(children):
            if i < fine_tune_at:
                for p in ch.parameters():
                    p.requires_grad = False
            else:
                for p in ch.parameters():
                    p.requires_grad = True

# --- LOSS / METRIKÁK ---
criterion = nn.BCEWithLogitsLoss()  # expects logits
# Helper to compute accuracy and AUC on probabilities
def evaluate_model(model, loader, device):
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            losses.append(loss.item() * imgs.size(0))
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs.tolist())
            targets.extend(labels.cpu().numpy().tolist())
    total = sum([len for len in [len(targets)]])  # trivial
    avg_loss = sum(losses) / max(1, len(targets))
    # accuracy threshold 0.5
    pred_labels = [1 if p > 0.5 else 0 for p in preds]
    acc = np.mean([pl == t for pl, t in zip(pred_labels, targets)])
    # AUC: if only one class present, roc_auc_score will fail; handle that.
    try:
        auc = roc_auc_score(targets, preds)
    except Exception:
        auc = float("nan")
    return avg_loss, acc, auc

# --- Callbacks: ModelCheckpoint + EarlyStopping (restore best weights) ---
class EarlyStopping:
    def __init__(self, monitor='val_loss', patience=4, restore_best_weights=True):
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best = None
        self.best_epoch = None
        self.num_bad = 0
        self.best_state = None
    def step(self, current, epoch, model):
        # lower is better for val_loss
        if self.best is None or current < self.best:
            self.best = current
            self.best_epoch = epoch
            self.num_bad = 0
            if self.restore_best_weights:
                self.best_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.num_bad += 1
            if self.num_bad > self.patience:
                return True
            return False
    def restore(self, model):
        if self.restore_best_weights and self.best_state is not None:
            model.load_state_dict(self.best_state)

# --- TRAINING LOOP FUNCTION ---
def train_phase(model, train_loader, val_loader, epochs, optimizer, scheduler=None,
                device=DEVICE, start_epoch=1, callbacks=None):
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    es = callbacks.get('earlystopping') if callbacks else None
    best_loss = float('inf')
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        running_loss = 0.0
        running_samples = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_samples += imgs.size(0)
        train_loss = running_loss / max(1, running_samples)

        # compute train acc quickly (on train_loader) - optional: do on subset to save time
        # Here compute on full train set for parity with TF behavior
        train_loss2, train_acc, _ = evaluate_model(model, train_loader, device)
        val_loss, val_acc, val_auc = evaluate_model(model, val_loader, device)

        history['loss'].append(train_loss2)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # checkpoint best by val_loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        # early stopping
        stop = False
        if es is not None:
            stop = es.step(val_loss, epoch, model)
            if stop:
                print(f"EarlyStopping triggered at epoch {epoch}")
                break

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss2:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_auc={val_auc:.4f}")

    return history

# --- 1) Első fázis: base frozen, lr=1e-4 ---
set_base_trainable(model, trainable=False)
# ensure head params trainable
for p in model.fc.parameters():
    p.requires_grad = True
for p in model.dropout.parameters():
    p.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
callbacks = {'earlystopping': es}

print("=== Training phase 1: head only ===")
h1 = train_phase(model, train_loader, val_loader, epochs=EPOCHS, optimizer=optimizer, device=DEVICE, start_epoch=1, callbacks=callbacks)

# restore best weights after phase 1 if EarlyStopping stored them; also best model saved to BEST_MODEL_PATH
if es.best_state is not None:
    model.load_state_dict(es.best_state)

# --- 2) Finomhangolás: unfreeze base, freeze first 100 layers (approx), lr=1e-5 ---
# make base trainable first
set_base_trainable(model, trainable=True)
# freeze first 100 modules (approx) - torchvision mobilenet_v2.features has many modules; we freeze first 100 children indexes if exist
fine_tune_at = 100
children = list(model.base.children())
for i, ch in enumerate(children):
    requires = False if i < fine_tune_at else True
    for p in ch.parameters():
        p.requires_grad = requires

# ensure classifier params trainable
for p in model.fc.parameters():
    p.requires_grad = True
for p in model.dropout.parameters():
    p.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# New EarlyStopping for phase 2 to track best in this phase too
es2 = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
callbacks2 = {'earlystopping': es2}

print("=== Fine-tuning phase 2: partial base trainable ===")
h2 = train_phase(model, train_loader, val_loader, epochs=EPOCHS//2, optimizer=optimizer, device=DEVICE, start_epoch=EPOCHS+1, callbacks=callbacks2)

# restore best weights from phase 2 if available
if es2.best_state is not None:
    model.load_state_dict(es2.best_state)
# also ensure we have final best saved file
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

# --- értékelés a teszten ---
test_loss, test_acc, test_auc = evaluate_model(model, test_loader, DEVICE)
print("Test:", test_loss, test_acc, test_auc)

# --- plot accuracy (combine h1 + h2) ---
acc = h1['accuracy'] + h2['accuracy']
val_acc = h1['val_accuracy'] + h2['val_accuracy']
epochs_range = range(1, len(acc) + 1)
plt.figure(figsize=(7,4))
plt.plot(epochs_range, acc, 'b-', label='train accuracy')
plt.plot(epochs_range, val_acc, 'r--', label='val accuracy')
plt.title("Training vs Validation Accuracy (PyTorch)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- 10 minta kép megjelenítése predikcióval ---
def show_sample_predictions(model, loader, n=10, class_names=None):
    model.eval()
    imgs = []
    trues = []
    preds = []
    with torch.no_grad():
        for batch_imgs, batch_labels in loader:
            for img, lab in zip(batch_imgs, batch_labels):
                imgs.append(img.cpu())
                trues.append(int(lab.item()))
                if len(imgs) >= n:
                    break
            if len(imgs) >= n:
                break
    # get predictions
    with torch.no_grad():
        batch = torch.stack([im for im in imgs]).to(DEVICE)
        logits = model(batch).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
    # un-normalize for display (we scaled to [-1,1])
    imgs_display = [((im + 1.0) / 2.0).permute(1,2,0).numpy() for im in imgs]
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols*3, rows*3))
    for i in range(len(imgs_display)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(np.clip(imgs_display[i], 0, 1))
        plt.axis('off')
        prob = probs[i]
        pred_label = class_names[1] if prob > 0.5 else class_names[0]
        true_label = class_names[trues[i]]
        title = f"P: {pred_label} ({prob:.2f})\nT: {true_label}"
        color = "green" if pred_label == true_label else "red"
        plt.title(title, color=color, fontsize=9)
    plt.tight_layout()
    plt.show()

show_sample_predictions(model, test_loader, n=10, class_names=class_names)