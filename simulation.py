import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 定义目标分布（高斯混合模型）
def target_distribution(num_samples):
    mean1 = torch.tensor([-1., 1.])
    cov1 = torch.tensor([[0.8, 0.], [0., 0.8]])
    dist1 = MultivariateNormal(mean1, cov1)
    samples1 = dist1.sample((num_samples // 2,))

    mean2 = torch.tensor([3., -3.])
    cov2 = torch.tensor([[0.8, 0.], [0., 0.8]])
    dist2 = MultivariateNormal(mean2, cov2)
    samples2 = dist2.sample((num_samples // 2,))

    combined_samples = torch.cat([samples1, samples2], dim=0)
    return combined_samples

# 定义只返回目标分布其中一支样本的函数
def target_distribution_single_branch(num_samples):
    mean1 = torch.tensor([-1., 1.])
    cov1 = torch.tensor([[0.8, 0.], [0., 0.8]])
    dist1 = MultivariateNormal(mean1, cov1)
    samples1 = dist1.sample((num_samples,))
    return samples1

# 定义具有完全可训练协方差矩阵的变分分布
class FullGaussianVariational(nn.Module):
    def __init__(self):
        super(FullGaussianVariational, self).__init__()
        self.mean = nn.Parameter(torch.tensor([-0.5, -0.5]))
        # 使用对数变换确保对角线元素为正
        self.cov_diag_log = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.cov_off_diag = nn.Parameter(torch.tensor([0.]))

    def forward(self):
        # 通过指数变换确保对角线元素为正
        cov_diag = torch.exp(self.cov_diag_log)
        # 构建下三角矩阵
        L = torch.stack([
            torch.stack([cov_diag[0], torch.tensor(0.)]),
            torch.stack([self.cov_off_diag[0], cov_diag[1]])
        ])
        # 通过 Cholesky 分解得到协方差矩阵
        cov_matrix = torch.matmul(L, L.t())
        return MultivariateNormal(self.mean, cov_matrix)

# 计算切片瓦瑟斯坦距离
def sliced_wasserstein_distance(mu_samples, nu_samples, num_slices=10):
    dim = mu_samples.shape[1]
    sw_distance = 0
    for _ in range(num_slices):
        theta = torch.randn(dim)
        theta = theta / torch.norm(theta)
        mu_proj = torch.matmul(mu_samples, theta)
        nu_proj = torch.matmul(nu_samples, theta)
        mu_proj, _ = torch.sort(mu_proj)
        nu_proj, _ = torch.sort(nu_proj)
        sw_distance += torch.sum(torch.abs(mu_proj - nu_proj))
    return sw_distance / num_slices

# 训练传统变分推断（VI）
def train_vi(target_dist, variational_dist, optimizer, num_epochs, num_samples):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        target_samples = target_dist(num_samples)
        target_dist_obj = MultivariateNormal(target_samples.mean(dim=0), torch.cov(target_samples.t()))
        variational_dist_obj = variational_dist()
        loss = torch.distributions.kl.kl_divergence(variational_dist_obj, target_dist_obj).sum()
        loss.backward()
        optimizer.step()
    return variational_dist

# 基于MCMC的SWVI训练
def train_swvi_mcmc(target_dist, variational_dist, optimizer, num_epochs, num_samples, num_markov_chains, lag, std_dev):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        target_samples = target_dist(num_samples)
        variational_dist_obj = variational_dist()
        variational_samples = variational_dist_obj.rsample((num_markov_chains,))  # 使用rsample保证可导性

        for _ in range(lag):
            # 提议步骤：随机游走
            proposal_state = variational_samples + std_dev * torch.randn_like(variational_samples)
            # 简单的接受准则（这里只是示例，实际可更复杂）
            current_loss = sliced_wasserstein_distance(variational_samples, target_samples)
            proposal_loss = sliced_wasserstein_distance(proposal_state, target_samples)
            acceptance = torch.exp(current_loss - proposal_loss) > torch.rand(1)
            variational_samples = torch.where(acceptance.unsqueeze(1), proposal_state, variational_samples)

        loss = sliced_wasserstein_distance(variational_samples, target_samples)
        loss.backward()
        optimizer.step()
    return variational_dist

# 改进后的EBSWVI方法
def train_ebswvi(target_dist, variational_dist, optimizer, num_epochs=15000, learning_rate=0.0003, num_samples=300, reg_lambda=0.000005):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.95)  # 调整学习率衰减步长
    huber_loss = nn.HuberLoss()  # 使用Huber损失

    for epoch in range(num_epochs):
        target_samples = target_dist(num_samples)
        variational_dist_obj = variational_dist()
        variational_samples = variational_dist_obj.rsample((num_samples,))

        output = variational_samples
        loss = huber_loss(output, target_samples)
        l2_reg = 0
        for param in variational_dist.parameters():
            l2_reg += torch.norm(param)
        loss = loss + reg_lambda * l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # 优化后验概率计算
    log_likelihood = -0.5 * ((variational_samples - target_samples) ** 2).sum()
    posterior_prob = torch.exp(log_likelihood - reg_lambda * l2_reg)
    return variational_dist

# 绘制等高线图
def plot_distributions(target_dist, vi_dist, swvi_dist, ebswvi_dist, num_samples=100000):  # 增加样本数量
    target_samples = target_dist(num_samples).numpy()
    vi_samples = vi_dist().sample((num_samples,)).numpy()
    swvi_samples = swvi_dist().sample((num_samples,)).numpy()
    ebswvi_samples = ebswvi_dist().sample((num_samples,)).numpy()

    xmin, xmax = -4, 6
    ymin, ymax = -6, 4
    # 提高网格分辨率
    X, Y = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # 目标分布等高线
    kernel_target = gaussian_kde(target_samples.T, bw_method=0.3)  # 手动调整带宽
    Z_target = np.reshape(kernel_target(positions).T, X.shape)
    plt.contour(X, Y, Z_target, levels=10, colors='g', linewidths=2)

    # VI分布等高线
    kernel_vi = gaussian_kde(vi_samples.T, bw_method=0.3)  # 手动调整带宽
    Z_vi = np.reshape(kernel_vi(positions).T, X.shape)
    plt.contour(X, Y, Z_vi, levels=10, colors='r', linewidths=2)

    # SWVI分布等高线
    kernel_swvi = gaussian_kde(swvi_samples.T, bw_method=0.3)  # 手动调整带宽
    Z_swvi = np.reshape(kernel_swvi(positions).T, X.shape)
    plt.contour(X, Y, Z_swvi, levels=10, colors='b', linewidths=2)

    # EBSWVI分布等高线
    kernel_ebswvi = gaussian_kde(ebswvi_samples.T, bw_method=0.3)  # 手动调整带宽
    Z_ebswvi = np.reshape(kernel_ebswvi(positions).T, X.shape)
    plt.contour(X, Y, Z_ebswvi, levels=10, colors='m', linewidths=2)

    # 手动创建图例条目
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='g', lw=2, label='Target Distribution'),
        Line2D([0], [0], color='r', lw=2, label='VI'),
        Line2D([0], [0], color='b', lw=2, label='SWVI'),
        Line2D([0], [0], color='m', lw=2, label='EBSW-VI')
    ]

    # 使用legend函数自动生成图例，并将其放置在右上角
    plt.legend(handles=legend_elements, loc='upper right')

    # plt.title('Approximation of Target Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    num_epochs = 300
    num_samples = 300
    learning_rate = 0.01
    num_markov_chains = 300
    lag = 20
    std_dev = 0.2

    # 初始化变分分布
    vi_dist = FullGaussianVariational()
    swvi_dist = FullGaussianVariational()
    ebswvi_dist = FullGaussianVariational()

    # 定义优化器
    vi_optimizer = optim.Adam(vi_dist.parameters(), lr=learning_rate)
    swvi_optimizer = optim.Adam(swvi_dist.parameters(), lr=learning_rate)
    ebswvi_optimizer = optim.Adam(ebswvi_dist.parameters(), lr=learning_rate)

    # 训练变分分布
    trained_vi_dist = train_vi(target_distribution_single_branch, vi_dist, vi_optimizer, num_epochs, num_samples)
    trained_swvi_dist = train_swvi_mcmc(target_distribution, swvi_dist, swvi_optimizer, num_epochs, num_samples,
                                        num_markov_chains, lag, std_dev)
    trained_ebswvi_dist = train_ebswvi(target_distribution, ebswvi_dist, ebswvi_optimizer, num_epochs)

    # 绘制分布
    plot_distributions(target_distribution, trained_vi_dist, trained_swvi_dist, trained_ebswvi_dist)