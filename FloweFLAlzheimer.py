import random
from typing import Dict

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

NUM_CLIENTS = 3
NUM_CLASSES = 3
LOCAL_EPOCHS = 3
ROUNDS = 80
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 5e-4
SEED = 1234
DATA_ROOT = "/Users/luanr/pycharm/FedAlzheimer/ADNI 3"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

_train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
_test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

full_ds = datasets.ImageFolder(DATA_ROOT)
targets = np.array(full_ds.targets)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20,
                             random_state=SEED)
train_idx, test_idx = next(sss.split(np.arange(len(full_ds)), targets))

train_subset = Subset(full_ds, train_idx)
test_subset = Subset(full_ds, test_idx)


def make_noniid_partitions(idxs: np.ndarray,
                           num_clients: int,
                           alpha: float = .5) -> Dict[int, np.ndarray]:
    y = targets[idxs]
    cid2idxs = {cid: np.empty(0, dtype=int) for cid in range(num_clients)}
    for cls in range(NUM_CLASSES):
        cls_mask = (y == cls)
        cls_idxs = idxs[cls_mask]
        rng.shuffle(cls_idxs)
        proportions = rng.dirichlet([alpha] * num_clients)
        split_pts = (np.cumsum(proportions) * len(cls_idxs)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(cls_idxs, split_pts)):
            cid2idxs[cid] = np.concatenate([cid2idxs[cid], chunk])
    return cid2idxs


client_train_partitions = make_noniid_partitions(train_idx, NUM_CLIENTS)
client_test_partitions = make_noniid_partitions(test_idx, NUM_CLIENTS)


class WrappedSubset(Dataset):

    def __init__(self, subset: Subset, tf):
        self.subset, self.tf = subset, tf

    def __getitem__(self, i):
        x, y = self.subset[i]
        return self.tf(x), y

    def __len__(self): return len(self.subset)


def build_model(num_classes=NUM_CLASSES) -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss, epoch_acc, seen = 0., 0., 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)
        epoch_acc += (y_hat.argmax(1) == y).sum().item()
        seen += x.size(0)
    return epoch_loss / seen, epoch_acc / seen


def test(model, loader, criterion):
    model.eval()
    loss, acc, preds, lbls, logits = 0., 0., [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = model(x)
            loss += criterion(y_hat, y).item() * x.size(0)
            acc += (y_hat.argmax(1) == y).sum().item()
            preds.extend(y_hat.argmax(1).cpu())
            lbls.extend(y.cpu())
            logits.append(F.softmax(y_hat, 1).cpu())
    loss /= len(loader.dataset)
    acc /= len(loader.dataset)
    logits = torch.cat(logits)
    f1 = torchmetrics.functional.f1_score(torch.tensor(preds),
                                          torch.tensor(lbls),
                                          task="multiclass",
                                          num_classes=NUM_CLASSES,
                                          average="macro").item()
    auc = torchmetrics.functional.auroc(logits,
                                        torch.tensor(lbls),
                                        task="multiclass",
                                        num_classes=NUM_CLASSES).item()
    return loss, acc, f1, auc

class TorchClient(fl.client.NumPyClient):

    def __init__(self,
                 cid: str,
                 train_idx: np.ndarray,
                 test_idx: np.ndarray):
        self.cid = int(cid)
        self.model = build_model().to(DEVICE)

        y_local = targets[train_idx]
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_local),
            y=y_local
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float).to(DEVICE)
        )

        self.train_loader = DataLoader(
            WrappedSubset(Subset(full_ds, train_idx), _train_tf),
            batch_size=BATCH_SIZE, shuffle=True
        )
        self.test_loader = DataLoader(
            WrappedSubset(Subset(full_ds, test_idx), _test_tf),
            batch_size=BATCH_SIZE, shuffle=False
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters, config):
        state_dict = dict(zip(self.model.state_dict().keys(),
                              [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.get("lr", LR),
            weight_decay=WEIGHT_DECAY
        )

        epochs = int(config.get("epochs", LOCAL_EPOCHS))
        for _ in range(epochs):
            train_one_epoch(self.model, self.train_loader,
                            self.criterion, optimizer)

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss, acc, f1, auc = test(self.model, self.test_loader, self.criterion)
        return float(loss), len(self.test_loader.dataset), {
            "acc": float(acc), "f1": f1, "auc": auc
        }


def get_evaluate_fn(test_loader_server):
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    def evaluate(server_round: int,
                 parameters: fl.common.NDArrays,
                 config):
        # Atualiza pesos globais
        state_dict = dict(zip(model.state_dict().keys(),
                              [torch.tensor(p) for p in parameters]))
        model.load_state_dict(state_dict, strict=True)

        loss, acc, f1, auc = test(model, test_loader_server, criterion)
        return loss, {"acc": acc, "f1": f1, "auc": auc}

    return evaluate


def main():
    test_loader_global = DataLoader(
        WrappedSubset(test_subset, _test_tf),
        batch_size=BATCH_SIZE, shuffle=False
    )

    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_evaluate_fn(test_loader_global),
        on_fit_config_fn=lambda rnd:
        {"epochs": LOCAL_EPOCHS, "lr": LR},
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    def client_fn(cid: str) -> TorchClient:
        return TorchClient(cid,
                           client_train_partitions[int(cid)],
                           client_test_partitions[int(cid)])

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(),
        strategy=strategy,
        client_resources={"num_cpus": 4, "num_gpus": 1 if torch.cuda.is_available() else 0},
    )


if __name__ == "__main__":
    main()
