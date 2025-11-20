# SFDFF 方法的代码

## 环境
> conda env create -f environment.yaml

## 模型

### 在 ImageNet 数据集上预训练的模型
>在运行程序的时候，以下模型可被自动下载。
 'ResNet18', 'ResNet50','vgg16','inception_v3','efficientnet_b0', 'DenseNet121', 'mobilenet_v2','inception_resnet_v2','inception_v4_timm','xception','vit_base_patch16_224','levit_384','convit_base','twins_svt_base','pit'

>请从 <https://github.com/microsoft/robust-models-transfer> 下载 “resnet50_l2_eps0_1” 的权重文件，并将其移动到以下路径： "./resnet50_l2_eps0.1.ckpt"。

### 在 CIFAR-10 数据集上预训练的模型
>请从 <https://github.com/huyvnphan/PyTorch_CIFAR10> 下载下列模型权重文件，并将其移动到以下路径 "./PyTorch_CIFAR10/cifar10_models/state_dicts/"
 
'vgg16_bn', 'resnet18', 'resnet50','mobilenet_v2','inception_v3','densenet121'

## 数据集
你可以从右边的链接中下载完整的 ImageNet-Compatible 数据集，一共1000张样本： https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset
下载完数据集后请将其移动至以下路径： './dataset/images'.

## 使用方法

你可以输入以下命令来直接执行攻击实验：

> python eval_attacks.py --config_idx=578



