import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import os
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'other': 0, 'smoking': 1}
        self.image_paths = []
        self.labels = []

        for cls_name in os.listdir(root_dir):
            cls_folder = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if cls_name == 'other':
                            label = self.classes['other']
                        else:
                            label = self.classes['smoking']

                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
     
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(train_dir, val_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BaseDataset(root_dir=train_dir, transform=transform)
    val_dataset = BaseDataset(root_dir=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset.classes


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(in_features=256 * 14 * 14, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # shape_features = 256x14x14
        x = x.view(-1, 256 * 14 * 14)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def init_model(device):
    model = BaseModel()
    model = model.to(device)
    return model


def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        print(f'Train learn: Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Loss: {epoch_loss:.4f}')

        model.eval()
        val_running_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            print('Valid')
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) >= 0.5
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        print(f'val_Loss: {val_loss:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}')

    return model


TRAIN_DIR = './data/train/'
VAL_DIR = './data/val/'


if __name__ == "__main__":
    train_loader, val_loader, classes = get_data_loaders(TRAIN_DIR, VAL_DIR)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_model(device)
    
    num_epochs = 1
    model = train_model(model, train_loader, val_loader, device, num_epochs)

    torch.save(model.state_dict(), 'baseline.pth')
