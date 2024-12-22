# @Time   : 2024/12/17
# @Author : yzh93
# @Email  :

from recbole.data.dataset import Dataset



class ColdStartDataset(Dataset):

# todo yzh  这里需要增加冷启动数据集相关的配置项
    def __init__(self,config):
        super(ColdStartDataset,self).__init__(config)
        pass
