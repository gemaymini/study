import torch
import os
import compareloss数据增强 as net
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import math

# 加载模型
model = net.SimCLRStage1(feature_dim=256)
model.load_state_dict(
    torch.load(r"导入模型文件路径", map_location=torch.device('cpu')))
model.eval()

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])


def extract_features(image_path, transform, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加 batch 维度
    with torch.no_grad():
        feature = model.f(image)
        feature = torch.flatten(feature, start_dim=1)
    return feature


def cosine_similarity(feature1, feature2):
    similarity = torch.nn.functional.cosine_similarity(feature1, feature2)
    return similarity.item()


def map_similarity(similarity):
    similarity = similarity+1
    similarity_mapped=similarity*50
    return similarity_mapped


def main(image_path1, image_path2):
    # 提取特征
    feature1 = extract_features(image_path1, transform, model)
    feature2 = extract_features(image_path2, transform, model)

    # 计算余弦相似度
    similarity = cosine_similarity(feature1, feature2)

    # 映射余弦相似度到[0, 1]范围
    similarity_mapped = map_similarity(similarity)

    return feature1, feature2, similarity_mapped


if __name__ == "__main__":
    image_path1_ = ["标准字路径列表"]

    image_path2_ = ["临摹字路径列表"]

    for i in range(len(image_path1_)):
        image_path1 = image_path1_[i]
        image_path2 = image_path2_[i]
        # 获取特征向量和余弦相似度
        feature1, feature2, similarity_mapped = main(image_path1, image_path2)
        print(f"Mapped Cosine Similarity:{similarity_mapped:.2f} ")

