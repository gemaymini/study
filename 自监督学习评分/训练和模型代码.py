import torch
import argparse
import os
from torchvision.models import resnet152
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import (Grayscale, ToTensor, RandomCrop, RandomHorizontalFlip, ColorJitter, RandomRotation,
                                    RandomAffine, Resize, Normalize)
from PIL import Image
import numpy as np
from torchvision.transforms.functional import perspective
from scipy.ndimage import gaussian_filter, map_coordinates
import random
import config
import loaddataset
import torch.nn.functional as F
# 比原版多了数据增强和adamw 可以调整adamw的学习率解决loss为nan
# 修改了数据增强 裁剪尺寸 扩大尺寸后内存不够
# 修改了adamw学习率 le-3 168
# 修改了loss的温度 0.3 165
# 能预训练
class RandomElasticTransform(object):
    def __init__(self, alpha_range=(0, 50), sigma_range=(4, 6), alpha_affine_range=(0, 0.1)):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.alpha_affine_range = alpha_affine_range

    def __call__(self, img):
        alpha = random.uniform(*self.alpha_range)
        sigma = random.uniform(*self.sigma_range)
        alpha_affine = random.uniform(*self.alpha_affine_range)
        random_state = np.random.RandomState(None)
        shape = img.size
        shape_size = shape[:2]
        blur_size = int(4 * sigma) | 1
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        distorted_image = map_coordinates(np.array(img), indices, order=1, mode='reflect')
        distorted_image = distorted_image.reshape(shape)
        return Image.fromarray(distorted_image.astype(np.uint8))


# 修改 transform_func 函数，添加数据增强操作
def transform_func(image):
    if image.mode == '1':
        image = Resize((100, 100))(image)
        image = RandomCrop((95, 95))(image)
        image = RandomRotation(5)(image)
        image = ToTensor()(image)
    else:
        image = Grayscale(num_output_channels=1)(image)
        image = Resize((100, 100))(image)
        image = RandomCrop((95, 95))(image)
        image = RandomRotation(5)(image)
        image = ToTensor()(image)
        image = ToTensor()(image)
    image = Normalize(mean=[0.5], std=[0.5])(image)
    return image


class SimCLRStage1(torch.nn.Module):
    def __init__(self, resnet_model=None, feature_dim=128, freeze_layers=6):
        super(SimCLRStage1, self).__init__()
        # 使用预训练的 ResNet-152 模型或者从头开始
        if resnet_model is None:
            self.f = resnet152(pretrained=True)
        else:
            self.f = resnet_model
        # 冻结前 freeze_layers 层
        if freeze_layers > 0:
            for param in self.f.parameters():
                param.requires_grad = False
            for param in self.f.layer4.parameters():
                param.requires_grad = True
        # 修改 ResNet-152 的最后一层，使其适应自监督学习任务
        num_ftrs = self.f.fc.in_features
        self.f.fc = torch.nn.Linear(num_ftrs, feature_dim)
        # projection head
        self.g = torch.nn.Sequential(torch.nn.Linear(feature_dim, 512, bias=False),
                                     torch.nn.BatchNorm1d(512),
                                     torch.nn.ReLU(inplace=True),
                                     torch.nn.Dropout(p=0.3),  # 添加 Dropout
                                     torch.nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        # 将单通道图像复制为三通道
        x = torch.cat([x, x, x], dim=1)
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return torch.nn.functional.normalize(feature, dim=-1), torch.nn.functional.normalize(out, dim=-1)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
        mask = torch.eye(len(z), device=sim_matrix.device)
        pos_sim = torch.diag(sim_matrix, len(z) // 2)
        neg_sim = (sim_matrix.sum(dim=1) - 2 * torch.sum(sim_matrix * mask, dim=1)) / (len(z) - 2)
        neg_sim = neg_sim[:len(z) // 2]
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)).mean()
        return loss


def train(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current device:", DEVICE)

    # 加载数据集
    dataset = loaddataset.PreDataset(root=config.datasetpath, transform=transform_func)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                             drop_last=True)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 定义模型，使用预训练的 ResNet-152 模型
    model = SimCLRStage1(feature_dim=256)

    # 检查是否已经保存了模型参数，如果没有则导入预训练模型
    model_path = "上次模型参数文件路径"
    if os.path.exists(model_path):
        print("Loading saved model parameters...")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # 获取当前模型的参数字典
        model_dict = model.state_dict()
        # 过滤掉不匹配的键
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 更新模型参数
        model_dict.update(state_dict)
        # 加载模型参数
        model.load_state_dict(model_dict)
        print("Loaded saved model parameters from:", model_path)
    else:
        print("No saved model parameters found. Training from scratch...")
    # 设置模型的参数为可训练
    for param in model.parameters():
        param.requires_grad = True
    # 将模型移动到设备上
    model.to(DEVICE)
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    # 定义学习率调整策略
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = ContrastiveLoss(temperature=0.3).to(DEVICE)  # 使用对比损失函数

    os.makedirs(config.save_path, exist_ok=True)
    best_val_loss = float('inf')
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        total_loss = 0
        total_batches = 0  # 统计总批次数
        for batch, (imgL, imgR) in enumerate(train_data):
            imgL, imgR = imgL.to(DEVICE), imgR.to(DEVICE)

            feature_L, _ = model(imgL)
            feature_R, _ = model(imgR)

            loss = loss_fn(feature_L, feature_R)  # 不需要传入 batch_size 参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch", epoch, "batch", batch, "loss:", loss.detach().item())
            total_loss += loss.detach().item()
            total_batches += 1

        epoch_loss = total_loss / total_batches
        print("epoch loss:", epoch_loss)

        with open("损失文件路径", "a") as f:
            f.write(str(epoch_loss) + " ")

        val_loss = evaluate(model, val_data, loss_fn, DEVICE)
        print("Validation Loss:", val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "最优模型参数路径")
            print("Save best model")

        # 更新学习率
        scheduler.step()


def evaluate(model, val_data, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        total_batches = 0  # 统计总批次数
        for imgL, imgR in val_data:
            imgL, imgR = imgL.to(device), imgR.to(device)
            feature_L, _ = model(imgL)
            feature_R, _ = model(imgR)
            loss = loss_fn(feature_L, feature_R)  # 不需要传入 batch_size 参数
            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=128, type=int, help='')
    parser.add_argument('--max_epoch', default=200, type=int, help='')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')

    args = parser.parse_args()
    train(args)