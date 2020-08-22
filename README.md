# AutoDict-CCL2020

- 论文: 基于BERT与柱搜索的中文释义生成 

- 联系方式: <blcufqn@hotmail.com>

## 环境配置

python (3.6)

numpy (1.19.1)

torch (1.4.0)

torchvision (0.2.1)

transformers (2.9.1)

nltk (3.5)

sklearn (0.0)

sentence_transformers (0.3.3)

## 数据准备

请从[此处](https://drive.google.com/drive/folders/1KwWxiRU_lOl9VcaCORBeKp0msCrEpkkY?usp=sharing)下载预训练中文词向量（merge_sgns_bigram_char300.txt）和英文词向量（cc.en.300.vec）文件，分别置于./data/cwn和./data/oxford目录下。

注：以下仅展示中文CWN数据集上的实验过程，英文实验过程相同。

```shell
cd ./data/cwn
python make_vocab.py
python make_vector.py
```

## 实验

**模型训练** 分为两个阶段

第一阶段：固定编码器参数，仅训练解码器。

```shell
./train_cbert01.sh 
```

第二阶段：同时调优编码器和解码器参数。注意将脚本中load_model参数的***改为上一阶段训练保存的最好模型。

```shell
./train_cbert02.sh
```

**模型测试**

```shell
./test_cbert01.sh
./test_cbert02.sh
python caculate_semantic_similarity.py # 计算语义相似度指标
```

**人工评价**

我们从CWN测试集中随机抽取了200个句子，请四名标注员对5个模型的生成结果从语义和语法两个角度进行独立评分，评价结果见human_evaluation目录。
