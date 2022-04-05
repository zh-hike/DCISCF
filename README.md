# Deep Convolutional Neural Network for Identifying Seam-Carving Forgery
> **author**: Hai Zhou \
> **date** : 2022.4.5

本次实验主要是使用pytorch复现[DCISCF](https://arxiv.org/abs/2007.02393) 论文。

## 准备
### 数据集
* 训练集在 `./dataset/data/train/`, 若要扩充数据集，直接将图片加载对应的文件夹里
* 验证集在 `./dataset/data/test/`, 若要扩充数据集，直接将图片加载对应的文件夹里

### 依赖库
**注意！！！！（默认在linux）**

如果想要运行代码，首先需要配置环境
```angular2html
conda create -n zh python=3.9
conda activate zh
python -m pip install --upgrade pip
pip install -r requirements.txt
```

最好使用在 *torch-gpu* 环境运行，需要运行如下
```angular2html
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

## 训练
* 训练命令

`python run.py --train`

* 验证命令

`python run.py --eval`
