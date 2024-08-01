from torchvision.datasets import VisionDataset
from torchvision import transforms
from PIL import Image
import os


class PreDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(PreDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for root, _, fnames in sorted(os.walk(self.root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                samples.append(path)
        return samples

    def __getitem__(self, index):
        path = self.samples[index]
        img = Image.open(path).convert('RGB')

        # 第一张图像的数据增强操作
        if self.transform is not None:
            imgL = self.transform(img)

        # 第二张图像的数据增强操作
        if self.transform is not None:
            imgR = self.transform(img)

        return imgL, imgR

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import config

    train_data = PreDataset(root=config.datasetpath, transform=config.train_transform)
    print(train_data[0])
