# @Time   : 2024/12/17
# @Author : yzh93
# @Email  :
from collections import defaultdict
from recbole.data.dataset import Dataset
from recbole.utils import (
    FeatureSource,
    set_color,
)

import numpy as np
import os
import pandas as pd

class ColdStartDataset(Dataset):
    """
    1. 直接根据config中配置读取数据集的配置，然后手动分割冷热数据集
    """
    def __init__(self, config):
        super(ColdStartDataset, self).__init__(config)

    def _data_processing(self):
        """Data preprocessing, including:

        - Data filtering
        - Missing value imputation
        - Normalization
        - Preloading weights initialization
        """
        self.feat_name_list = self._build_feat_name_list()
        if self.benchmark_filename_list is None:
            self._data_filtering()

        self._user_item_feat_preparation()
        self._fill_nan()
        self._set_label_by_threshold()
        self._normalize()
        self._preload_weight_matrix()

    def _load_and_update_data(self, dataset_name, path):
        """
        该方法每次加载数据并更新当前对象中的属性
        """
        data = self._load_data(dataset_name, path)

        # 动态创建属性并赋值
        data_attr_name = path.split('/')[-2] + '_data'  # 从路径中提取数据集名称并加上 '_data'
        setattr(self, data_attr_name, data)

        # 在此可以根据需要进行其他操作
        self.logger.info(f"Loaded data from {path} and assigned to {data_attr_name}")

    def _get_field_from_config(self):
        """Initialization common field names."""
        self.uid_field = self.config["USER_ID_FIELD"]
        self.iid_field = self.config["ITEM_ID_FIELD"]
        self.label_field = self.config["LABEL_FIELD"]
        self.time_field = self.config["TIME_FIELD"]

        if (self.uid_field is None) ^ (self.iid_field is None):
            raise ValueError(
                "USER_ID_FIELD and ITEM_ID_FIELD need to be set at the same time or not set at the same time."
            )

        self.logger.debug(set_color("uid_field", "blue") + f": {self.uid_field}")
        self.logger.debug(set_color("iid_field", "blue") + f": {self.iid_field}")

    def _load_data(self, token, dataset_path):
        """Load features.

        Firstly load interaction features, then user/item features optionally,
        finally load additional features if ``config['additional_feat_suffix']`` is set.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        inter_feat = self._load_inter_feat(token, dataset_path)
        self.user_feat = self._load_user_or_item_feat(
            token, dataset_path, FeatureSource.USER, "uid_field"
        )
        self.item_feat = self._load_user_or_item_feat(
            token, dataset_path, FeatureSource.ITEM, "iid_field"
        )
        return inter_feat

    def _load_inter_feat(self, token, dataset_path):
        """Load interaction features.

        If ``config['benchmark_filename']`` is not set, load interaction features from ``.inter``.

        Otherwise, load interaction features from a file list, named ``dataset_name.xxx.inter``,
        where ``xxx`` if from ``config['benchmark_filename']``.
        After loading, ``self.file_size_list`` stores the length of each interaction file.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        if self.benchmark_filename_list is None:
            inter_feat_path = os.path.join(dataset_path, f"{token}.inter")
            if not os.path.isfile(inter_feat_path):
                raise ValueError(f"File {inter_feat_path} not exist.")

            inter_feat = self._load_feat(inter_feat_path, FeatureSource.INTERACTION)
            self.logger.debug(
                f"Interaction feature loaded successfully from [{inter_feat_path}]."
            )
            self.inter_feat = inter_feat
        else:
            sub_inter_lens = []
            sub_inter_feats = []
            overall_field2seqlen = defaultdict(int)
            for filename in self.benchmark_filename_list:
                file_path = os.path.join(dataset_path, f"{token}.{filename}.inter")
                if os.path.isfile(file_path):
                    temp = self._load_feat(file_path, FeatureSource.INTERACTION)
                    sub_inter_feats.append(temp)
                    sub_inter_lens.append(len(temp))
                    for field in self.field2seqlen:
                        overall_field2seqlen[field] = max(
                            overall_field2seqlen[field], self.field2seqlen[field]
                        )
                else:
                    raise ValueError(f"File {file_path} not exist.")
            inter_feat = pd.concat(sub_inter_feats, ignore_index=True)
            self.inter_feat, self.file_size_list = inter_feat, sub_inter_lens
            self.field2seqlen = overall_field2seqlen
            return inter_feat

    # def build(self):
    #     """Processing dataset according to evaluation setting, including Group, Order and Split.
    #     See :class:`~recbole.config.eval_setting.EvalSetting` for details.
    #
    #     Returns:
    #         list: List of built :class:`Dataset`.
    #     """
    #     self._change_feat_format()
    #
    #     if self.benchmark_filename_list is not None:
    #         self._drop_unused_col()
    #         cumsum = list(np.cumsum(self.file_size_list))
    #         datasets = [
    #             self.copy(self.inter_feat[start:end])
    #             for start, end in zip([0] + cumsum[:-1], cumsum)
    #         ]
    #         return datasets
    #
    #     # ordering
    #     ordering_args = self.config["eval_args"]["order"]
    #     if ordering_args == "RO":
    #         self.shuffle()
    #
    #     return datasets






