import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

# 定义多项式回归模型
class PolynomialRegression(nn.Module):
    def __init__(self, degree):
        super(PolynomialRegression, self).__init__()
        self.degree = degree
        self.fc = nn.Linear(degree + 1, 1)

    def forward(self, x):
        poly_features = [x ** i for i in range(self.degree + 1)]
        poly_features = torch.cat(poly_features, dim=1)
        return self.fc(poly_features)

# 模拟数据生成函数
def generate_data(degree, num_samples=10000, noise_std=0.01):  # 增加数据量
    x = torch.linspace(0, 1, num_samples).unsqueeze(1)
    y = torch.sum(torch.randn(degree + 1) * x ** torch.arange(degree + 1), dim=1).unsqueeze(1)
    y += torch.randn_like(y) * noise_std
    return x, y

# 改进后的EBSWVI方法
def EBSWVI(model, x, y, num_epochs=15000, learning_rate=0.0003, num_mc_samples=100, reg_lambda=0.000005):  # 调整超参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.95)  # 调整学习率衰减步长
    huber_loss = nn.HuberLoss()  # 使用Huber损失

    for epoch in range(num_epochs):
        output = model(x)
        loss = huber_loss(output, y)
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = loss + reg_lambda * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # 优化后验概率计算
    log_likelihood = -0.5 * ((model(x) - y) ** 2).sum()
    posterior_prob = torch.exp(log_likelihood - reg_lambda * l2_reg)
    return posterior_prob

# 改进后的SWVI方法
def SWVI(model, x, y, num_epochs=10000, learning_rate=0.0005, num_mc_samples=100):  # 调整超参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)  # 调整学习率衰减步长
    for epoch in range(num_epochs):
        output = model(x)
        mse_loss = torch.mean((output - y) ** 2)
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        scheduler.step()

    log_likelihood = -0.5 * ((model(x) - y) ** 2).sum()
    # 考虑模型复杂度惩罚
    num_params = sum(p.numel() for p in model.parameters())
    posterior_prob = torch.exp(log_likelihood - 0.00001 * num_params)  # 调整正则化参数
    return posterior_prob

# 改进后的VI方法
def VI(model, x, y, num_epochs=10000, learning_rate=0.0005, num_mc_samples=100):  # 调整超参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)  # 调整学习率衰减步长
    for epoch in range(num_epochs):
        output = model(x)
        mse_loss = torch.mean((output - y) ** 2)
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        scheduler.step()

    log_likelihood = -0.5 * ((model(x) - y) ** 2).sum()
    # 考虑模型复杂度惩罚
    num_params = sum(p.numel() for p in model.parameters())
    posterior_prob = torch.exp(log_likelihood - 0.00001 * num_params)  # 调整正则化参数
    return posterior_prob

# 实验主函数
def experiment(num_trials=100, num_degrees=5, num_samples=10000, noise_std=0.01):  # 增加数据量
    correct_count_ebswvi = 0
    correct_count_swvi = 0
    correct_count_vi = 0

    for trial in range(num_trials):
        true_degree = random.randint(1, num_degrees)
        x, y = generate_data(true_degree, num_samples, noise_std)

        best_ebswvi_degree = None
        best_ebswvi_prob = -1
        best_swvi_degree = None
        best_swvi_prob = -1
        best_vi_degree = None
        best_vi_prob = -1

        for degree in range(1, num_degrees + 1):
            model = PolynomialRegression(degree)
            ebswvi_posterior = EBSWVI(model, x, y)
            if ebswvi_posterior > best_ebswvi_prob:
                best_ebswvi_prob = ebswvi_posterior
                best_ebswvi_degree = degree

            model = PolynomialRegression(degree)
            swvi_posterior = SWVI(model, x, y)
            if swvi_posterior > best_swvi_prob:
                best_swvi_prob = swvi_posterior
                best_swvi_degree = degree

            model = PolynomialRegression(degree)
            vi_posterior = VI(model, x, y)
            if vi_posterior > best_vi_prob:
                best_vi_prob = vi_posterior
                best_vi_degree = degree

        if best_ebswvi_degree == true_degree:
            correct_count_ebswvi += 1
        if best_swvi_degree == true_degree:
            correct_count_swvi += 1
        if best_vi_degree == true_degree:
            correct_count_vi += 1

    accuracy_ebswvi = correct_count_ebswvi / num_trials
    accuracy_swvi = correct_count_swvi / num_trials
    accuracy_vi = correct_count_vi / num_trials

    print(f"EBSWVI选择正确模型的概率: {accuracy_ebswvi}")
    print(f"SWVI选择正确模型的概率: {accuracy_swvi}")
    print(f"VI选择正确模型的概率: {accuracy_vi}")

    # 绘制柱状图
    methods = ['EBSW-VI', 'SWVI', 'VI']
    accuracies = [accuracy_ebswvi, accuracy_swvi, accuracy_vi]

    plt.bar(methods, accuracies)
    # plt.xlabel('Methods')
    plt.ylabel('Accuracy')
    # plt.title('Accuracy of Model Selection Methods')
    plt.show()

if __name__ == "__main__":
    experiment()