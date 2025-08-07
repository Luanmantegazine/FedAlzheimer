import flwr as fl, torch
from model import ResNet50FL
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

NUM_CLIENTS = 3
ROUNDS = 80
FRACTION = 1.0


def get_eval_fn():
    root = "/content/drive/MyDrive/ADNI"
    full_ds = datasets.ImageFolder(root=root,
                                   transform=transforms.Compose([
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
                                   ]))
    test_idx = sum(json.load(open("partitions/indices.json"))["test"].values(), [])
    test_ds = torch.utils.data.Subset(full_ds, test_idx)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50FL(num_classes=3).to(device)

    def evaluate(server_round, parameters, config):
        st = model.state_dict()
        for k, v in zip(st.keys(), parameters):
            st[k] = torch.tensor(v, dtype=st[k].dtype)
        model.load_state_dict(st)
        model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss += torch.nn.functional.cross_entropy(logits, y,
                                                          reduction='sum').item()
                correct += (logits.argmax(1) == y).sum().item()
        num_ex = len(test_ds)
        return loss / num_ex, {"accuracy": correct / num_ex}

    return evaluate


strategy = fl.server.strategy.FedAvg(
    fraction_fit=FRACTION,
    eval_fn=get_eval_fn(),
    on_fit_config_fn=lambda r: {"local_epochs": 3},
)

fl.simulation.start_simulation(
    client_fn=lambda cid: AlzheimerClient(cid),
    num_clients=NUM_CLIENTS,
    strategy=strategy,
    server_config=fl.server.ServerConfig(num_rounds=ROUNDS),
    client_resources={"num_gpus": 1 if torch.cuda.is_available() else 0}
)