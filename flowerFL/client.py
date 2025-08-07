import flwr as fl, torch, torch.nn.functional as F, torchmetrics, json, numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.utils.class_weight import compute_class_weight
from model import ResNet50FL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LOCAL_EPOCHS = 3
LR = 1e-4
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 3

# Transforms
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_partition(client_id: int):
    parts = json.load(open("partitions/indices.json"))
    root = "/content/drive/MyDrive/ADNI"
    full_ds = datasets.ImageFolder(root=root)
    tr_idx = parts["train"][str(client_id)]
    te_idx = parts["test"][str(client_id)]
    train_ds = Subset(full_ds, tr_idx)
    test_ds = Subset(full_ds, te_idx)
    return train_ds, test_ds


class AlzheimerClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = int(cid)
        self.model = ResNet50FL(num_classes=NUM_CLASSES).to(DEVICE)
        self.train_ds, self.test_ds = load_partition(self.cid)

        # Class‑weighting (recalcula a cada cliente)
        labels = [self.train_ds.dataset.targets[i] for i in self.train_ds.indices]
        cw = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(cw, dtype=torch.float).to(DEVICE)
        )

    # Auxiliares ----------------------------------------------------------------
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        st = self.model.state_dict()
        for k, v in zip(st.keys(), parameters):
            st[k] = torch.tensor(v, dtype=st[k].dtype)
        self.model.load_state_dict(st)

    # Fit -----------------------------------------------------------------------
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR,
                                     weight_decay=WEIGHT_DECAY)
        loader = DataLoader(self.train_ds, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=lambda x: (
                torch.stack([train_tf(i[0]) for i in x]),
                torch.tensor([i[1] for i in x])
            ))
        for _ in range(LOCAL_EPOCHS):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward();
                optimizer.step()
        return self.get_parameters(), len(self.train_ds), {}

    # Evaluate ------------------------------------------------------------------
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        test_loader = DataLoader(self.test_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, collate_fn=lambda x: (
                torch.stack([test_tf(i[0]) for i in x]),
                torch.tensor([i[1] for i in x])
            ))
        all_preds, all_labels, all_logits, loss_sum = [], [], [], 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = self.model(x)
                loss_sum += self.criterion(logits, y).item() * x.size(0)
                all_logits.append(torch.softmax(logits, 1).cpu())
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        num_ex = len(self.test_ds)
        preds = torch.tensor(all_preds);
        lbls = torch.tensor(all_labels)
        logits = torch.cat(all_logits)
        metrics = {
            "accuracy": torchmetrics.functional.accuracy(
                preds, lbls, task="multiclass", num_classes=NUM_CLASSES
            ).item(),
            "f1_macro": torchmetrics.functional.f1_score(
                preds, lbls, average="macro", task="multiclass", num_classes=NUM_CLASSES
            ).item(),
            "auc_macro": torchmetrics.functional.auroc(
                logits, lbls, task="multiclass", num_classes=NUM_CLASSES
            ).item(),
        }
        return loss_sum / num_ex, num_ex, metrics
