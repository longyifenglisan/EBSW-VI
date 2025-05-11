import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import pandas as pd

# 定义贝叶斯逻辑回归模型，修改为多分类
class BayesianLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(BayesianLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 10)  # 10 个类别

    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)

# VI方法训练
def vi_train(model, dataloader, num_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for X, y in dataloader:
            X = X.view(X.size(0), -1)
            y = F.one_hot(y, num_classes=10).float()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, torch.argmax(y, dim=1))
            loss.backward()
            optimizer.step()
    return model

# SWVI方法训练，使用改进的MCMC采样策略（ESWVI）
def eswvi_train(model, dataloader, num_chains, num_steps, step_size, lag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = list(model.parameters())
    all_chains = []
    for _ in range(num_chains):
        chain = []
        for t in range(num_steps):
            for X, y in dataloader:
                X = X.view(X.size(0), -1)
                y = F.one_hot(y, num_classes=10).float()
                if t < lag:
                    # 预热阶段，仅进行MCMC采样
                    for i, param in enumerate(params):
                        grad = torch.autograd.grad(nn.CrossEntropyLoss()(
                            model(X), torch.argmax(y, dim=1)), [param])[0]
                        param.data += step_size * (grad + torch.randn_like(param) * np.sqrt(2 * step_size))
                else:
                    # 正式训练阶段，更新模型参数
                    new_params = []
                    for i, param in enumerate(params):
                        grad = torch.autograd.grad(nn.CrossEntropyLoss()(
                            model(X), torch.argmax(y, dim=1)), [param])[0]
                        new_param = param.data + step_size * (grad + torch.randn_like(param) * np.sqrt(2 * step_size))
                        new_params.append(new_param)
                    new_model = BayesianLogisticRegression(X.shape[1]).to(device)
                    for i, param in enumerate(new_model.parameters()):
                        param.data = new_params[i]
                    chain.append(new_model)
        all_chains.append(chain)
    # 选择最优模型（例如，基于验证集性能）
    best_model = None
    best_acc = 0
    for chain in all_chains:
        for sub_model in chain:
            total_correct = 0
            total_samples = 0
            for X, y in dataloader:
                X = X.view(X.size(0), -1)
                y = F.one_hot(y, num_classes=10).float()
                with torch.no_grad():
                    outputs = sub_model(X)
                    predicted = torch.argmax(outputs, dim=1)
                    total_correct += (predicted == torch.argmax(y, dim=1)).sum().item()
                    total_samples += y.size(0)
            acc = total_correct / total_samples
            if acc > best_acc:
                best_acc = acc
                best_model = sub_model
    return best_model

# 改进后的EBSWVI方法，加入早停法和L1正则化
def EBSWVI(model, dataloader, num_epochs=1000, learning_rate=0.0005, reg_lambda=0.5, patience=30):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    counter = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for X, y in dataloader:
            X = X.view(X.size(0), -1)
            y = F.one_hot(y, num_classes=10).float()
            output = model(X)
            bce_loss = criterion(output, torch.argmax(y, dim=1))

            reg_loss = 0
            for param in model.parameters():
                reg_loss += torch.norm(param, p=1)

            loss = bce_loss + reg_lambda * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    return model

# 评估模型
def evaluate(model, dataloader):
    total_correct = 0
    total_samples = 0
    for X, y in dataloader:
        X = X.view(X.size(0), -1)
        y = F.one_hot(y, num_classes=10).float()
        with torch.no_grad():
            outputs = model(X)
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == torch.argmax(y, dim=1)).sum().item()
            total_samples += y.size(0)
    accuracy = total_correct / total_samples
    return accuracy

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# 创建数据加载器
dataloader_train = DataLoader(train_subset, batch_size=100, shuffle=True)
dataloader_val = DataLoader(val_subset, batch_size=100, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=100, shuffle=True)

# 初始化模型
input_dim = 28 * 28

# VI模型训练与评估
vi_model = BayesianLogisticRegression(input_dim)
vi_model = vi_train(vi_model, dataloader_train, num_epochs=100, lr=0.001)
vi_accuracy = evaluate(vi_model, dataloader_test)
print(f"VI Test Accuracy: {vi_accuracy:.4f}")

# SWVI模型训练与评估
swvi_model = BayesianLogisticRegression(input_dim)
swvi_model = eswvi_train(swvi_model, dataloader_train, num_chains=10, num_steps=50, step_size=0.0001, lag=10)
swvi_accuracy = evaluate(swvi_model, dataloader_test)
print(f"SWVI Test Accuracy: {swvi_accuracy:.4f}")

# EBSWVI模型训练与评估（L1正则化）
eswvi_model_l1 = BayesianLogisticRegression(input_dim)
eswvi_model_l1 = EBSWVI(eswvi_model_l1, dataloader_train, num_epochs=200, learning_rate=0.001, reg_lambda=0.01, patience=50)
eswvi_accuracy_l1 = evaluate(eswvi_model_l1, dataloader_test)
print(f"EBSW-VI Test Accuracy: {eswvi_accuracy_l1:.4f}")

# # 创建一个 DataFrame 来保存结果
# results = {
#     'Model': ['VI', 'SWVI', 'EBSW-VI'],
#     'Test Accuracy': [vi_accuracy, swvi_accuracy, eswvi_accuracy_l1]
# }
# df = pd.DataFrame(results)
#
# # 将 DataFrame 保存为 Excel 文件
# df.to_excel('model_accuracies.xlsx', index=False)