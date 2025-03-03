# 实验计划
## 框架的有效性验证
### 模型有效性验证
1. 通过跑相同数据集ml-100k，验证在BPR LightGCN SGL NGCF下的指标效果
   设置：300 epoch 其他设置一样，早停设置10

## 实验数据的分割
1. 所有的数据按照7:3 进行分割 热 冷数据集
2. 对于热数据不进行处理，进行拆分，5 训练，1 验证 1测试
3. 对于冷数据进行1 训练 1 验证 1 测试的处理
4. 这里要注意，需要将在 验证和测试集中有而在训练集中没有的数据都转移到训练集中
   1. 这里的数据集有ml-100k CUL ml-1m
## 实验指标验证
这里在训练的时候，不要用配置文件中的设置，而是自动加载训练集，验证集，测试集，然后进行测试。
先在warm上进行验证，这里已经验证了两个不同的基准数据集CUL和ml-1m
nohup python run_recbole.py --model=LightGCN --port=6777  --dataset=CiteULike >> CiteULike_LGNwarm.log 2>&1 &
nohup python run_recbole.py --model=BPR --port=6778  --dataset=CiteULike >> CiteULike_BPRwarm.log 2>&1 &
nohup python run_recbole.py --model=SGL --port=6779  --dataset=CiteULike >> CiteULike_SGLwarm.log 2>&1 &
nohup python run_recbole.py --model=NGCF --port=6780  --dataset=CiteULike >> CiteULike_NGCFwarm.log 2>&1 &

