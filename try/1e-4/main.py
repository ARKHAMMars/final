import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

num_epochs = 300
batch_size = 128
learning_rate = 0.0001  
weight_decay = 1e-4  
momentum = 0.9
num_trials = 3

gamma = 0.9
beta = 0.999
epsilon = 1e-8
lambda_ = 5e-4  

# 数据增强配置
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 自定义Adan优化器
class AdanOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, gamma=0.9, beta=0.999, epsilon=1e-8, lambda_=0.01, weight_decay=0):
        defaults = dict(lr=lr, gamma=gamma, beta=beta, epsilon=epsilon, lambda_=lambda_, weight_decay=weight_decay)
        super(AdanOptimizer, self).__init__(params, defaults)
        
        # 初始化m_t和v_t
        self.m = {}
        self.v = {}

    def step(self, closure=None):
        # 执行优化器步骤
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            beta = group['beta']
            epsilon = group['epsilon']
            lambda_ = group['lambda_']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # 初始化m_t和v_t
                if p not in self.m:
                    self.m[p] = torch.zeros_like(p.data)
                    self.v[p] = torch.zeros_like(p.data)
                
                # 计算一阶矩和二阶矩
                m_t = gamma * self.m[p] + (1 - gamma) * grad
                v_t = beta * self.v[p] + (1 - beta) * grad.pow(2)
                
                # 计算更新公式
                p.data = p.data - lr / (torch.sqrt(v_t) + epsilon) * (gamma * m_t + grad)
                
                ## 添加标准权重衰减项：按AdamW权重衰减
                p.data.add_(-group['lr'] * group['lambda_'], p.data)
                
                # 更新m_t和v_t
                self.m[p] = m_t
                self.v[p] = v_t
        
        return loss


# 获取数据加载器
def get_dataloaders():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    return trainloader, testloader


def train_one_epoch(model, trainloader, criterion, optimizer, device, scheduler=None, warmup_scheduler=None):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(trainloader, desc='Training'):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()

        if scheduler:
            scheduler.step()
        if warmup_scheduler:
            warmup_scheduler.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return train_loss / len(trainloader), 100 * correct / total


def test_model(model, testloader, criterion, device):
    model.eval()
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            correct += outputs.argmax(1).eq(labels).sum().item()
    
    return test_loss / len(testloader), 100 * correct / len(testloader.dataset)


def train_model(trial_id, device):
    writer = SummaryWriter(log_dir=f'runs/resnet18_trial_{trial_id}')
    trainloader, testloader = get_dataloaders()

    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()

    # 使用自定义的 AdanOptimizer
    optimizer = AdanOptimizer(model.parameters(), lr=learning_rate, lambda_=weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1
    )
    
    first_80_epoch, first_90_epoch = None, None
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, scheduler, warmup_scheduler)
        test_loss, test_acc = test_model(model, testloader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if first_80_epoch is None and test_acc >= 80:
            first_80_epoch = epoch + 1
        if first_90_epoch is None and test_acc >= 90:
            first_90_epoch = epoch + 1
        
        writer.add_scalars('Loss', {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'test': test_acc}, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
        
        print(f'Trial {trial_id} | Epoch {epoch+1:03d}: '
              f'LR {current_lr:.4f} | Test Acc {test_acc:.2f}%')
    
    writer.close()
    
    return test_acc, first_80_epoch, first_90_epoch


# 训练多个实验
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

accuracies = []
first_80_epochs = []
first_90_epochs = []
for trial in range(num_trials):
    torch.cuda.empty_cache()
    acc, first_80_epoch, first_90_epoch = train_model(trial, device)
    accuracies.append(acc)
    first_80_epochs.append(first_80_epoch)
    first_90_epochs.append(first_90_epoch)

print(f'ResNet-18 Final Accuracies: {accuracies}')
print(f'Mean: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%')
print(f'First Epoch to Reach 80% Accuracy in Each Trial: {first_80_epochs}')
print(f'First Epoch to Reach 90% Accuracy in Each Trial: {first_90_epochs}')
