# MAGTKD
Jie Li, Shifei Ding, Lili Guo, and Xuan Li, "Multi-modal Anchor Gated Transformer with Knowledge Distillation for Emotion Recognition in Conversation". (IJCAI 2025, Pytorch Code)

## Abstract
Emotion Recognition in Conversation (ERC) aims to detect the emotions of individual utterances within a conversation. Generating efficient and modality-specific representations for each utterance remains a significant challenge. Previous studies have proposed various models to integrate features extracted using different modality-specific encoders. However, they neglect the varying contributions of modalities to this task and introduce high complexity by aligning modalities at the frame level. To address these challenges, we propose the Multi-modal Anchor Gated Transformer with Knowledge Distillation (MAGTKD) for the ERC task. Specifically, prompt learning is employed to enhance textual modality representations, while knowledge distillation is utilized to strengthen representations of weaker modalities. Furthermore, we introduce a multi-modal anchor gated transformer to effectively integrate utterance-level representations across modalities. Extensive experiments on the IEMOCAP and MELD datasets demonstrate the effectiveness of knowledge distillation in enhancing modality representations and achieve state-of-the-art performance in emotion recognition. Our code is available at: https://github.com/JieLi-dd/MAGTKD.

<picture>
<img src="./src/Framework.jpg" width="1000" height="500">
</picture>

## Requirements
我们需要准备的用来提取三个模态特征的预训练模型的特征分别是：
文本模态采用[RoBERTa-large](https://huggingface.co/FacebookAI/roberta-large)进行训练，
语音模态采用[data2vec-audio-base-960h](https://huggingface.co/facebook/data2vec-audio-base-960h)进行训练，
视频模态的特征提取采用[Videomae-base](https://huggingface.co/MCG-NJU/videomae-base)进行提取。
视频模态采用[timesformer-base-finetuned-k400](https://huggingface.co/facebook/timesformer-base-finetuned-k400)进行训练。

我们需要的Python环境如下：
```
python==3.9.19
torch==1.13.1+cu116
torchvision==0.14.1+cu116   
torchaudio==0.13.1+cu116
transformers==4.27.2
```


## Datasets
我们两个广泛使用的ERC数据集分别是[IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)和[MELD](https://github.com/declare-lab/MELD/)数据集，原始的数据集可以根据超链接进行下载。
原始的数据集在第一阶段通过蒸馏学习进行不同模态特征的提取。我们也提供了第一阶段采用蒸馏学习提取好的特征。可以通过链接进行下载。
```
原始特征放入到下面的文件夹中：
Project
| - datasets
    | - IEMOCAP
        | - IEMOCAP_train.csv
        | - IEMOCAP_dev.csv
        | - IEMOCAP_test.csv
        | - Session1
        ...
    | - MELD
        | - train_meld_emo.csv
        | - dev_meld_emo.csv
        | - test_meld_emo.csv
        | - dev_splits_complete
        | - train_splits
        | - output_repeated_splits_test
        ...
| - pretrained_model
    | - roberta-large
    | - data2vec-audio-base-960h
    | - timesformer-base-finetuned-k400
    | - videomae-base
| - IEMOCAP
    | - feature
        | - video
            | - train
            | - dev
            | - test
    | - IEMOCAP
        | - save_model
            | - text.bin
            | - audio.bin
            | - video.bin
            | - text_KD_audio.bin
            | - video_KD_text.bin
    | - model.py
    | - utils.py
    | - preprocessing.py
    | - dataset.py
    | - text.py
    | - audio.py
    | - video.py
    | - video_feature_extract.py
    | - KD.py
| - MELD
```

## Train and test
对于从开始进行训练，请按照以下步骤进行：

对于IEMOCAP数据集：
```
1. 运行text.py文件获取文本模态的特征
python text.py
2. 运行audio.py文件获取音频模态的特征
python audio.py
3. 运行video_feature_extract.py文件获取视频模态的初始特征
python video_feature_extract.py 
4. 运行video.py文件获取视频模态的特征
python video.py
5. 运行KD.py文件进行 Knowledge Distillation获取增强的音频模态特征
python KD.py --student video --teacher text
6. 运行KD.py文件进行 Knowledge Distillation获取增强的视频模态特征
python KD.py --student video --teacher text
7. 运行extract_first_stage_features.py文件提取三个模态训练好的特征
python extract_first_stage_features.py
8. 运行multimodal_fusion.py文件进行多模态融合
python multimodal_fusion.py
```

对于MELD数据集：其结构以及运行过程和IEMOCAP数据集一样。

如果打算只是测试我们的模型的结果，请首先将IEMOCAP和MELD数据集第一阶段的特征文件下载保存到对应数据集的文件夹之下：
[baidu](https://pan.baidu.com/s/1t3Y1jdWgMXqhCkaT6gB1ww?pwd=dzz5),
[Google]()
```
python multimodal_fusion.py --train True
```

## Citation
如果您觉得我们的方法有用，请考虑引用我们的论文：
```
@inproceedings{song-etal-2022-supervised,
    title = "Supervised Prototypical Contrastive Learning for Emotion Recognition in Conversation",
    author = "Song, Xiaohui  and
      Huang, Longtao  and
      Xue, Hui  and
      Hu, Songlin",
    booktitle = "EMNLP",
    year = "2022",
    pages = "5197--5206",
}
@inproceedings{yun-etal-2024-telme,
    title = "{T}el{ME}: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation",
    author = "Yun, Taeyang  and
      Lim, Hyunkuk  and
      Lee, Jeonghwan  and
      Song, Min",
    booktitle = "NAACL",
    year = "2024",
    pages = "82--95",
}
@ARTICLE{10109845,
  author={Ma, Hui and Wang, Jian and Lin, Hongfei and Zhang, Bo and Zhang, Yijia and Xu, Bo},
  journal={IEEE Transactions on Multimedia}, 
  title={A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations}, 
  year={2024},
  volume={26},
  number={},
  pages={776-788},
}
```


## Acknowledgement
我们的方法在[SPCL](https://github.com/caskcsg/spcl)、[TelME](https://github.com/yuntaeyang/TelME)、[SDT](https://github.com/butterfliesss/SDT)方法的基础上运行。
感谢这些方法的作者提供的代码。