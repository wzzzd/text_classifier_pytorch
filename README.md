# Text_Classifier_Pytorch

## Info
基于Pytorch的文本分类框架。

同时支持中英文的数据集的实体识别任务。


## 支持的模型
- FastText
- TextCNN
- TextRNN
- TextRCNN
- Transformer
- Bert
- Albert
- Roberta
- Distilbert
- Electra
- XLNet                                  



## 支持的训练模式

- 支持中英文语料训练
    - 支持中英文的信息抽取任务
- 混合精度训练
    - 用于提升训练过程效率，缩短训练时间
    - 配置文件`Config.py`中的变量`adv_option`
- GPU多卡训练
    - 用于分布式训练，支持单机单卡、多卡训练
    - 配置文件`Config.py`中的变量`cuda_visible_devices`用于设置可见的GPU卡号，多卡情况下用`,`间隔开
- 对抗训练
    - 在模型embedding层增加扰动，使模型学习对抗扰动，提升表现，需要额外增加训练时间
    - 配置文件`Config.py`中的变量`adv_option`用于设置可见的对抗模式，目前支持FGM/PGD
- 对比学习
    - 用于增强模型语义特征提取能力，待补充


## Requirement
```
    python3.6
    numpy==1.19.5
    pandas==1.1.3
    torch==1.8.0
    transformers==4.6.1
    apex==0.1 (安装方法见：https://github.com/NVIDIA/apex)
```

可通过以下命令安装依赖包
```
    pip install -r requirement.txt
```

## Datasets
* **THUCNews**
    * 来自：https://github.com/649453932/Chinese-Text-Classification-Pytorch
    * 关于THUCNews的的数据。
    * 数据分为10个类标签类别，分别为：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐

* **加入自己的数据集**
    * 可使用本项目的处理方式，将数据集切分为3部分：train/valid/test，其中token和label之间用制表符`\t`分割。
    * 在 ./dataset 目录下新建一个文件夹，并把3个数据文件放置新建文件夹下。

* **数据集示例**
    * 以数据集THUCNews为栗子，文本和标签使用空格隔开，采用以下形式存储：
    ```

    ```


## Get Started
### 1. 训练
准备好训练数据后，终端可运行命令
```
    python3 main.py
```
### 2 测试评估
加载已训练好的模型，并使用valid set作模型测试，输出文件到 ./dataset/${your_dataset}/output/output.txt 目录下。

需要修改Config文件中的变量值`mode = 'test'`，并保存。

终端可运行命令
```
    python3 main.py
```


## Result
模型预测结果示例如下：

指标情况
![指标](./file/metrics.png)

测试集
![指标](./file/predict.png)


## FAQ


## Reference

【Github:transformers】https://github.com/huggingface/transformers



