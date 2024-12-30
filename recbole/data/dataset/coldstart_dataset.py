# @Time   : 2024/12/17
# @Author : yzh93
# @Email  :

from recbole.data.dataset import Dataset



class ColdStartDataset(Dataset):
    """
    1. 根据数据集中的冷启动数据分离配置进行冷热数据生成
        - 热：冷 = 8:2
        - 热数据 train：val：test = 8:1:1
        - 冷数据 val：test = 1:1
    """
# todo yzh  这里需要增加冷启动数据集相关的配置项
    def __init__(self,config):
        super().__init__(config)
        self._split_warm_cold(config) #根据config中的数据分割配置进行数据分割
