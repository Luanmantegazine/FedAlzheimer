from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np, torch, random, os, json

SEED = 1234
random.seed(SEED);
np.random.seed(SEED);
torch.manual_seed(SEED)

data_path = "/content/drive/MyDrive/ADNI"
num_clients = 3
num_classes = 3

transform_basic = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_ds = datasets.ImageFolder(root=data_path)
train_idx, test_idx = train_test_split(
    np.arange(len(full_ds)),
    test_size=0.20,
    stratify=full_ds.targets,
    random_state=SEED
)


def dataset_noniid(indices, num_clients, alpha=0.5):
    y = np.array(full_ds.targets)[indices]
    idx = np.arange(len(indices))
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_clients)}
    for c in range(num_classes):
        idx_c = idx[y == c]
        np.random.shuffle(idx_c)
        props = np.random.dirichlet(np.repeat(alpha, num_clients))
        split_pts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(idx_c, split_pts)):
            dict_users[i] = np.concatenate((dict_users[i], chunk))
    return dict_users


dict_users_train = dataset_noniid(train_idx, num_clients)
dict_users_test = dataset_noniid(test_idx, num_clients)

os.makedirs("partitions", exist_ok=True)
json.dump({
    "train": {k: v.tolist() for k, v in dict_users_train.items()},
    "test": {k: v.tolist() for k, v in dict_users_test.items()}
}, open("partitions/indices.json", "w"))
