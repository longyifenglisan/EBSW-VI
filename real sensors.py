import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import urllib.error
import time
import mask

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载数据集并进行预处理，以Heart数据集为例，Wine和Ionosphere数据集处理方式类似
def load_and_preprocess_data(data_name):
    max_retries = 3
    retry_delay = 5  # 重试间隔时间（秒）

    def download_data(url):
        retries = 0
        while retries < max_retries:
            try:
                df = pd.read_csv(url, header=None)
                return df
            except urllib.error.HTTPError as e:
                if e.code == 502:
                    print(f"HTTP Error 502: Bad Gateway. Retrying in {retry_delay} seconds...")
                    retries += 1
                    time.sleep(retry_delay)
                else:
                    raise
        raise Exception("Failed to download data after multiple retries.")

    if data_name == 'Heart':
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        df = download_data(url)
        df = df.replace('?', np.nan).dropna()
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)
        y = np.where(y > 0, 1, 0)
    elif data_name == 'Wine':
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
        df = download_data(url)
        X = df.iloc[:, 1:].values.astype(np.float32)
        y = df.iloc[:, 0].values.astype(np.float32)
        # 选取前两个类别
        selected_classes = np.unique(y)[:2]
        mask = np.isin(y, selected_classes)
        X = X[mask]
        y = y[mask]
        # 处理Wine数据集的目标值，使其在0到1之间
        class_mapping = {cls: idx for idx, cls in enumerate(selected_classes)}
        y = np.array([class_mapping[cls] for cls in y]).astype(np.float32)
    elif data_name == 'Ionosphere':
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        df = download_data(url)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values
        y = np.where(y == 'g', 1, 0).astype(np.float32)  # 确保y为float32类型
    else:
        raise ValueError('Invalid data name')

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    dataset = TensorDataset(X, y)
    return dataset

# 定义贝叶斯逻辑回归模型
class BayesianLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(BayesianLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# VI方法训练
def vi_train(model, X, y, num_epochs, lr):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs.squeeze(1), y)
        loss.backward()
        optimizer.step()
    return model

# SWVI方法训练，使用改进的MCMC采样策略（ESWVI）
def eswvi_train(model, X, y, num_chains, num_steps, step_size, lag):
    params = list(model.parameters())
    all_chains = []
    for _ in range(num_chains):
        chain = []
        for t in range(num_steps):
            if t < lag:
                # 预热阶段，仅进行MCMC采样
                for i, param in enumerate(params):
                    grad = torch.autograd.grad(torch.nn.functional.binary_cross_entropy_with_logits(
                        model(X), y.unsqueeze(1)), [param])[0]
                    param.data += step_size * (grad + torch.randn_like(param) * np.sqrt(2 * step_size))
            else:
                # 正式训练阶段，更新模型参数
                new_params = []
                for i, param in enumerate(params):
                    grad = torch.autograd.grad(torch.nn.functional.binary_cross_entropy_with_logits(
                        model(X), y.unsqueeze(1)), [param])[0]
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
            with torch.no_grad():
                outputs = sub_model(X)
                predicted = (outputs > 0.5).squeeze(1).float()
                acc = (predicted == y).sum().item() / y.size(0)
                if acc > best_acc:
                    best_acc = acc
                    best_model = sub_model
    return best_model

# 改进后的EBSWVI方法，加入早停法和L1正则化
def EBSWVI(model, X, y, num_epochs=1000, learning_rate=0.0005, reg_lambda=0.5, patience=30):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)
    criterion = nn.BCELoss()
    best_loss = float('inf')
    counter = 0
    for epoch in range(num_epochs):
        output = model(X)
        bce_loss = criterion(output.squeeze(1), y)

        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.norm(param, p=1)

        loss = bce_loss + reg_lambda * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    return model

# 评估模型
def evaluate(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predicted = (outputs > 0.5).squeeze(1).float()
        accuracy = (predicted == y).sum().item() / y.size(0)
    return accuracy

# 实验主函数
def run_experiment(data_name):
    dataset = load_and_preprocess_data(data_name)
    X = dataset.tensors[0]
    y = dataset.tensors[1]
    input_dim = X.shape[1]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    vi_accuracies = []
    swvi_accuracies = []
    eswvi_accuracies_l1 = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # VI模型训练与评估
        vi_model = BayesianLogisticRegression(input_dim).to(device)
        vi_model = vi_train(vi_model, X_train, y_train, num_epochs=50, lr=0.0001)
        vi_accuracy = evaluate(vi_model, X_test, y_test)
        vi_accuracies.append(vi_accuracy)

        # SWVI模型训练与评估
        swvi_model = BayesianLogisticRegression(input_dim).to(device)
        swvi_model = eswvi_train(swvi_model, X_train, y_train, num_chains=500, num_steps=150, step_size=0.00005, lag=30)
        swvi_accuracy = evaluate(swvi_model, X_test, y_test)
        swvi_accuracies.append(swvi_accuracy)

        # EBSWVI模型训练与评估（L1正则化）
        eswvi_model_l1 = BayesianLogisticRegression(input_dim).to(device)
        eswvi_model_l1 = EBSWVI(eswvi_model_l1, X_train, y_train)
        eswvi_accuracy_l1 = evaluate(eswvi_model_l1, X_test, y_test)
        eswvi_accuracies_l1.append(eswvi_accuracy_l1)

    vi_avg_accuracy = np.mean(vi_accuracies)
    swvi_avg_accuracy = np.mean(swvi_accuracies)
    eswvi_avg_accuracy_l1 = np.mean(eswvi_accuracies_l1)

    return vi_avg_accuracy, swvi_avg_accuracy, eswvi_avg_accuracy_l1

# 重复实验20次并取平均值
data_names = ['Heart', 'Wine', 'Ionosphere']
num_repeats = 1
all_vi_accuracies = {data_name: [] for data_name in data_names}
all_swvi_accuracies = {data_name: [] for data_name in data_names}
all_eswvi_accuracies_l1 = {data_name: [] for data_name in data_names}

for _ in range(num_repeats):
    for data_name in data_names:
        vi_acc, swvi_acc, eswvi_acc_l1 = run_experiment(data_name)
        all_vi_accuracies[data_name].append(vi_acc)
        all_swvi_accuracies[data_name].append(swvi_acc)
        all_eswvi_accuracies_l1[data_name].append(eswvi_acc_l1)

# 计算平均值
vi_avg_accuracies = [np.mean(all_vi_accuracies[data_name]) for data_name in data_names]
swvi_avg_accuracies = [np.mean(all_swvi_accuracies[data_name]) for data_name in data_names]
eswvi_avg_accuracies_l1 = [np.mean(all_eswvi_accuracies_l1[data_name]) for data_name in data_names]

# 打印平均值结果
for data_name in data_names:
    print(f"Dataset: {data_name}")
    print(f"VI Average Test Accuracy: {np.mean(all_vi_accuracies[data_name]):.4f}")
    print(f"SWVI Average Test Accuracy: {np.mean(all_swvi_accuracies[data_name]):.4f}")
    print(f"EBSW-VI Average Test Accuracy: {np.mean(all_eswvi_accuracies_l1[data_name]):.4f}")

# 绘制柱状图
x = np.arange(len(data_names))
width = 0.2
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in [('VI', vi_avg_accuracies), ('SWVI', swvi_avg_accuracies), ('EBSW-VI', eswvi_avg_accuracies_l1)]:
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    multiplier += 1

ax.set_ylabel('Accuracy')
# ax.set_title('Average Accuracy of Different Methods on Datasets (Repeated 10 times)')
ax.set_xticks(x + width, data_names)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 保存柱状图为PNG格式
plt.savefig('accuracy_bar_chart.png')

# 将20次重复实验输出的VI、SWVI、EBSW-VI的Accuracy保存到Excel格式的文件
data = {
    'Dataset': [],
    'VI Accuracy': [],
    'SWVI Accuracy': [],
    'EBSW-VI Accuracy': []
}

for data_name in data_names:
    for i in range(num_repeats):
        data['Dataset'].append(data_name)
        data['VI Accuracy'].append(all_vi_accuracies[data_name][i])
        data['SWVI Accuracy'].append(all_swvi_accuracies[data_name][i])
        data['EBSW-VI Accuracy'].append(all_eswvi_accuracies_l1[data_name][i])

df = pd.DataFrame(data)
df.to_excel('accuracy_results.xlsx', index=False)

plt.show()