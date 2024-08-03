import torch
from model import resnet152
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据
img_path = "待识别的汉字图片"
img = Image.open(img_path)

# 把图片变为三通道
if img.mode != 'RGB':
    img = img.convert('RGB')

img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

# 读取分类文件
try:
    with open("分类文件", 'r') as json_file:
        class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# 创建模型
model = resnet152(num_classes=100)
# 加载模型权重
model_weight_path = "模型权重文件"
model.load_state_dict(torch.load(model_weight_path))
model.eval()

# 识别汉字
with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

# 输出结果
predicted_class = class_indict.get(str(predict_cla), "Unknown class")
prediction_confidence = predict[predict_cla].item()
print(f"Predicted class: {predicted_class}, Confidence: {prediction_confidence:.4f}")
