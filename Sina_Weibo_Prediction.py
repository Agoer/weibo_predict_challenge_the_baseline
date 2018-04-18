#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = 'linjiexin'
import os
import time

class Model(object):

    def __init__(self):
        self.path = os.getcwd()
        train_file = 'data/weibo_train_data.txt'
        predict_data_file = 'data/weibo_predict_data.txt'
        self.result_file = 'data/weibo_result_data.txt'
        self.train_data_path = "%s/%s" % (self.path, train_file)
        self.predict_data_path = "%s/%s" % (self.path, predict_data_file)

    def fit(self, train_list):
        max_precision = 0
        best_f = 0; best_c = 0; best_l = 0
        for line in set(train_list):
            numerator = 0   #分子
            denominator = 0 #分母
            line_dict = line.split(',')
            i = int(line_dict[0])
            j = int(line_dict[1])
            k = int(line_dict[2])
            for record in train_list:
                record_dict = record.split(',')
                fr = int(record_dict[0])
                cr = int(record_dict[1])
                lr = int(record_dict[2])
                df = float(abs(i - fr) / (fr + 5.0))
                dc = float(abs(j - cr) / (cr + 3.0))
                dl = float(abs(k - lr) / (lr + 3.0))
                precision_i = 1 - 0.5 * df - 0.25 * dc - 0.25 * dl
                count_i = fr + cr + lr
                count_i = 100 if count_i > 100 else count_i
                sgn_i = 0.8 if precision_i > 0.8 else 0.0
                denominator += count_i + 1
                numerator += (count_i + 1) * sgn_i
            if numerator / denominator > max_precision:
                max_precision = numerator / denominator
                best_f = i
                best_c = j
                best_l = k
        return "%s,%s,%s" % (best_f, best_c, best_l)


    def get_history_fcl(self):
        map = {}
        with open(self.train_data_path) as f:
            for line in f.readlines():
                line_dict = line.split('\t')
                uid = line_dict[0]
                forcast_comment_like = "%s,%s,%s" % (line_dict[3], line_dict[4], line_dict[5])
                if uid not in map:
                    map[uid] = []
                map[uid].append(forcast_comment_like)
        return map

    def train_model(self):
        model = {}
        map = self.get_history_fcl()
        for uid in map:
            model[uid] = self.fit(map[uid])
        return model

    def output_result(self):
        model = self.train_model()
        result = []
        with open(self.predict_data_path) as f:
            for line in f.readlines():
                line_dict = line.split('\t')
                uid = line_dict[0]
                mid = line_dict[1]
                predict = model[uid] if uid in model else "0,0,0"
                result.append("%s\t%s\t%s\n" % (uid, mid, predict))
        with open(self.result_file, 'w+') as f:
            f.writelines(result)


if __name__ == '__main__':
    start_time = time.time()
    Model().output_result()
    end_time = time.time()
    print ("total cost time: %s seconds" % int(end_time - start_time))

