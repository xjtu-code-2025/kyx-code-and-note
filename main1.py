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

import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import *



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
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

# class own_dataset(Dataset):
#     # init,len,getitem
#     def __init__(self, root, preprocess):
#         super(own_dataset, self).__init__()
#         self.preprocess = preprocess
#         self.image_paths = []
#         self.labels = []
#         label_list = os.listdir(root)
#         for label in label_list:
#             # root=/home label=apple os.path.join(root,label):/home/apple
#             image_folder = os.path.join(root, label)
#             for file in os.listdir(image_folder):
#                 if file.endswith(("png", "jpg", "gif")):
#                     self.image_paths.append(os.path.join(image_folder, file))
#                     self.labels.append(label_list.index(label))
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, item):
#         image = Image.open(self.image_paths[item])
#         image = self.preprocess(image)
#         label = self.labels[item]
#         return image, label
#
#     def print_len(self):
#         print(len(self.image_paths))
#
#
# trainset = own_dataset(root="./data", preprocess=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
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
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)  # loss=L+\lambda||w||^2
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


for epoch in range(start_epoch, start_epoch + 20):
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