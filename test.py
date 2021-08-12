
from torchvision.datasets import CIFAR100


cifarTrainDataset=CIFAR100(root='../data',
                                train=True,
                                download=True
                                )
sample=cifarTrainDataset[0]
print(sample[1])