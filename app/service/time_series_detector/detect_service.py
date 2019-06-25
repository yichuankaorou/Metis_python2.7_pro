#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Tencent is pleased to support the open source community by making Metis available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import time
import os
import threading
from app.dao.time_series_detector import anomaly_op
from app.dao.time_series_detector import sample_op
from app.dao.time_series_detector import train_op
from time_series_detector.algorithm import xgboosting
from time_series_detector import detect
from app.common.errorcode import *
from app.common.common import *
from time_series_detector.common.tsd_errorcode import *
MODEL_PATH = os.path.join(os.path.dirname(__file__), './model/')


class DetectService(object):

    def __init__(self):
        self.sample_op_obj = sample_op.SampleOperation()
        self.anomaly_op_obj = anomaly_op.AbnormalOperation()
        self.detect_obj = detect.Detect()

    def __generate_model(self, data, task_id):
        """
        Start train a model

        :param data: Training dataset.This is a list and data such as below:
                data -> samples_list=[{"flag": x, "data": "346", "353", "321", ...},
                                      {"flag": y, "data": "346", "353", "321", ...},
                                      {"flag": z, "data": "346", "353", "321", ...},
                                       ......
                                    ]
        :param task_id: The id of the training task.
        """
        xgb_obj = xgboosting.XGBoosting()
        # pylint: disable=unused-variable
        # jizhi 调用 xgboost 算法模型，
        #       传输参数：
        #           task_id - 时间戳
        #           data - 样本数据集
        #       返回值含义：
        #           ret_code: 执行正确or错误码
        ret_code, ret_data = xgb_obj.xgb_train(data, task_id)
        current_timestamp = int(time.time())
        # jizhi 初始化数据库的链接
        train_op_obj = train_op.TrainOperation()
        if ret_code == 0:
            train_status = "complete"
            params = {
                "task_id": task_id,
                "end_time": current_timestamp,
                "status": train_status,
                "model_name": task_id + "_model"
            }
        else:
            train_status = "failed"
            params = {
                "task_id": task_id,
                "end_time": current_timestamp,
                "status": train_status,
                "model_name": ""
            }
        # jizhi 到此，模型训练成功，在表 train_task 更新训练完成的模型信息
        train_op_obj.update_model_info(params)

    def process_train(self, data):
        """
        Start a process to train model
        :param data: Training dataset.
        """
        sample_params = {
            "trainOrTest": data["trainOrTest"],
            "positiveOrNegative": data["positiveOrNegative"],
            "source": data["source"],
            "beginTime": data["beginTime"],
            "endTime": data["endTime"]
        }
        # jizhi 依据页面选择的样本信息，从数据库中获取样本数据
        # jizhi 调用dao.time_series_detector.sample_op.SampleOperation.sample_query_all()函数，
        #       依据前端选择的训练集和时间信息从数据库 metis 对应的 sample_dataset 表中抽取数据，返回 sample_list
        samples = self.sample_op_obj.sample_query_all(sample_params)
        # jizhi 返回的数据列表 samples 内容形式：
        #       samples -> samples_list[{"flag": "0", "data": "660,719,649 ...", 642,758,777 ...", "698,644,691 ..."},
        #                               {"flag": "1", "data": "660,719,649 ...", 642,758,777 ...", "698,644,691 ..."},
        #                               {"flag": "0", "data": "660,719,649 ...", 642,758,777 ...", "698,644,691 ..."},
        #                                ......
        #                              ]
        # jizhi 初始化训练模型函数 app.dao.time_series_detector.TrainOperation(), 就是初始化连接数据库
        train_op_obj = train_op.TrainOperation()
        samples_list = []
        positive_count = 0
        negative_count = 0
        # jizhi samples 是返回的样本数据列表
        for index in samples:
            # jizhi map函数将str列表转化成int字典值value
            samples_list.append({"flag": index["flag"], "data": map(int, index["data"].split(','))})
            # jizhi 统计正负样本数量
            if index["flag"] == 1:
                positive_count = positive_count + 1
            else:
                negative_count = negative_count + 1
        # jizhi round 函数实现对当前时间戳 time.time() 浮点值扩大 1000 倍的四舍五入值
        task_id = str(int(round(time.time() * 1000)))
        train_params = {
            "begin_time": int(time.time()),
            "end_time": int(time.time()),
            "task_id": task_id,
            "status": "running",
            "source": data["source"],
            "sample_num": len(samples_list),
            "postive_sample_num": positive_count,
            "negative_sample_num": negative_count
        }
        if positive_count == 0 or negative_count == 0:
            return build_ret_data(LACK_SAMPLE, "")
        # jizhi 插入数据到 metis 数据库的表 train_task 中，状态是 running
        train_op_obj.insert_train_info(train_params)
        try:
            # jizhi 到此完成数据准备，开始调用算法进行模型训练
            # jizhi 传入参数，samples_list 处理好的数据和 task_id 时间戳
            # jizhi 算法计算时，是一次性传入全部 A B C 的数据
            t = threading.Thread(target=self.__generate_model, args=(samples_list, task_id, ))
            t.setDaemon(False)
            t.start()
        except Exception:
            train_status = "failed"
            params = {
                "task_id": task_id,
                "end_time": int(time.time()),
                "status": train_status,
                "model_name": ""
            }
            # jizhi 训练模型失败，更新模型信息
            train_op_obj.update_model_info(params)
        return build_ret_data(OP_SUCCESS, "")

    def __list_is_digit(self, data):
        for index in data:
            try:
                float(index)
            except ValueError:
                return False
        return True

    def __check_param(self, data):
        if ("viewName" not in data.keys()) or ("viewId" not in data.keys()) or ("attrId" not in data.keys()) or ("attrName" not in data.keys()) or ("time" not in data.keys()) or ("dataC" not in data.keys()) or ("dataB" not in data.keys()) or ("dataA" not in data.keys()):
            return CHECK_PARAM_FAILED, "missing parameter"
        return OP_SUCCESS, ""

    def value_predict(self, data):
        ret_code, ret_data = self.__check_param(data)
        if ret_code != OP_SUCCESS:
            return build_ret_data(ret_code, ret_data)
        ret_code, ret_data = self.detect_obj.value_predict(data)
        if ret_code == TSD_OP_SUCCESS and ret_data["ret"] == 0:
            anomaly_params = {
                "view_id": data["viewId"],
                "view_name": data["viewName"],
                "attr_id": data["attrId"],
                "attr_name": data["attrName"],
                "time": data["time"],
                "data_c": data["dataC"],
                "data_b": data["dataB"],
                "data_a": data["dataA"]
            }
            self.anomaly_op_obj.insert_anomaly(anomaly_params)
        return build_ret_data(ret_code, ret_data)

    def rate_predict(self, data):
        ret_code, ret_data = self.__check_param(data)
        if ret_code != OP_SUCCESS:
            return build_ret_data(ret_code, ret_data)
        ret_data, ret_data = self.detect_obj.rate_predict(data)
        if ret_code == TSD_OP_SUCCESS and ret_data["ret"] == 0:
            anomaly_params = {
                "view_id": data["viewId"],
                "view_name": data["viewName"],
                "attr_id": data["attrId"],
                "attr_name": data["attrName"],
                "time": data["time"],
                "data_c": data["dataC"],
                "data_b": data["dataB"],
                "data_a": data["dataA"]
            }
            self.anomaly_op_obj.insert_anomaly(anomaly_params)
        return build_ret_data(OP_SUCCESS, ret_data)
