"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import academictorrents as at
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import *
import random
from torchvision.models import resnet18




class my_dataset(Dataset):
    def __init__(self, path, preprocess):
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        label_list = os.listdir(path)
        for label in label_list:
            image_folder = os.path.join(path, label)
            for file_names in os.listdir(image_folder):
                if file_names.endswith(("png", "jpg", "jpeg")):
                    self.image_paths.append(os.path.join(image_folder, file_names))
                    self.labels.append(label_list.index(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = self.preprocess(image)
        label = self.labels[item]
        return image, label


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



from torch.utils.data import Dataset
from PIL import Image
import os

class own_dataset(Dataset):
    def __init__(self, image_paths, labels, preprocess, label_map=None):
        self.preprocess = preprocess
        self.image_paths = image_paths
        self.labels = labels
        self.label_map = label_map if label_map is not None else {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        image = self.preprocess(image)
        label = self.labels[item]
        return image, label

    def print_len(self):
        print(len(self.image_paths))

    def print_label_map(self):
        print(self.label_map)


def split_dataset(root, preprocess, train_ratio=0.8, seed=42):
    random.seed(seed)
    all_images_by_class = {}
    label_map = {}
    label_idx = 0

    for file in os.listdir(root):
        if file.endswith(("png", "jpg", "jpeg", "gif")):
            # 提取“最后一个下划线前”的部分作为类别名
            label_name = file.rsplit('_', 1)[0]
            if label_name not in label_map:
                label_map[label_name] = label_idx
                label_idx += 1
            full_path = os.path.join(root, file)
            all_images_by_class.setdefault(label_name, []).append(full_path)

    train_image_paths, train_labels = [], []
    test_image_paths, test_labels = [], []

    for label_name, img_list in all_images_by_class.items():
        random.shuffle(img_list)
        split = int(len(img_list) * train_ratio)
        label_id = label_map[label_name]
        train_image_paths += img_list[:split]
        train_labels += [label_id] * split
        test_image_paths += img_list[split:]
        test_labels += [label_id] * (len(img_list) - split)

    trainset = own_dataset(train_image_paths, train_labels, preprocess, label_map)
    testset = own_dataset(test_image_paths, test_labels, preprocess, label_map)
    return trainset, testset
trainset,_= split_dataset(root="./petdata/images/images", preprocess=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)
_,testset = split_dataset(root="./petdata/images/images", preprocess=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)


# Model
print('==> Building model..')
# net = VGG('VGG19')
net = resnet18(pretrained=True)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
# 冻结除了最后一层之外的所有参数
for param in net.parameters():
    param.requires_grad = False

# 替换最后一层为数据集的输出（37类）
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 37)  # 仅这一层参数是可训练的




net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    # weight = net.state_dict()
    # torch.save(weight, "/your/path")
    # weight = torch.load("/your/path")
    # net.load_state_dict(weight)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# 只将最后一层参数交给优化器
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_losses = []
train_accs = []
test_losses = []
test_accs = []
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(trainloader, unit="batch", dynamic_ncols=True) as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # for param in net.parameters():
            #     print(param.data,param.grad)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # tqdm替换progressbar
            tepoch.set_postfix(loss=train_loss/(batch_idx+1), 
                               accuracy=100.*correct/total)
    #记录训练中每次循环的平均损失和准确率
    avg_loss = train_loss / len(trainloader)
    accuracy = 100. * correct / total
    train_losses.append(avg_loss)
    train_accs.append(accuracy)
def test(epoch):
    global best_acc
    net.eval()
    # for param in net.parameters():
    #     param.requires_grad = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(testloader, unit="batch") as tepoch:
            tepoch.set_description(f"Test {epoch}")
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tepoch.set_postfix(loss=test_loss/(batch_idx+1), 
                                   accuracy=100.*correct/total)
    #记录测试中每次循环的平均损失和准确率
    avg_loss = test_loss / len(testloader)
    accuracy = 100. * correct / total
    test_losses.append(avg_loss)
    test_accs.append(accuracy)
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    scheduler.step()

# 绘制训练和测试损失及准确率曲线
plt.figure(figsize=(12, 5))
    
# 绘制损失
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
    
# 绘制正确率
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()