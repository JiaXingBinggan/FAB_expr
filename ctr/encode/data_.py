import csv
import collections
import operator
from csv import DictReader
from datetime import datetime
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from itertools import islice
import random
import numpy as np

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def to_time_frac(hour, min, time_frac_dict):
    for key in time_frac_dict[hour].keys():
        if key[0] <= min <= key[1]:
            return str(time_frac_dict[hour][key])

def to_libsvm_encode(datapath, sample_type, time_frac_dict):
    print('###### to libsvm encode ######\n')
    oses = ["windows", "ios", "mac", "android", "linux"]
    browsers = ["chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie"]

    f1s = ["weekday", "hour", "IP", "region", "city", "adexchange", "domain", "slotid", "slotwidth", "slotheight",
           "slotvisibility", "slotformat", "creative", "advertiser"]

    f1sp = ["useragent", "slotprice"]

    f2s = ["weekday,region"]

    def featTrans(name, content):
        content = content.lower()
        if name == "useragent":
            operation = "other"
            for o in oses:
                if o in content:
                    operation = o
                    break
            browser = "other"
            for b in browsers:
                if b in content:
                    browser = b
                    break
            return operation + "_" + browser
        if name == "slotprice":
            price = int(content)
            if price > 100:
                return "101+"
            elif price > 50:
                return "51-100"
            elif price > 10:
                return "11-50"
            elif price > 0:
                return "1-10"
            else:
                return "0"

    def getTags(content):
        if content == '\n' or len(content) == 0:
            return ["null"]
        return content.strip().split(',')[:5]

    # initialize
    namecol = {}
    featindex = {}
    maxindex = 0

    fi = open(datapath + 'train.bid.all.csv', 'r')

    first = True

    featindex['truncate'] = maxindex
    maxindex += 1

    for line in fi:
        s = line.split(',')
        if first:
            first = False
            for i in range(0, len(s)):
                namecol[s[i].strip()] = i
                if i > 0:
                    featindex[str(i) + ':other'] = maxindex
                    maxindex += 1
            continue
        for f in f1s:
            col = namecol[f]
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in featindex:
                featindex[feat] = maxindex
                maxindex += 1
        for f in f1sp:
            col = namecol[f]
            content = featTrans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in featindex:
                featindex[feat] = maxindex
                maxindex += 1
        # col = namecol["usertag"]
        # tags = getTags(s[col])
        # # for tag in tags:
        # feat = str(col) + ':' + ''.join(tags)
        # if feat not in featindex:tian
        #     featindex[feat] = maxindex
        #     maxindex += 1

    print('feature size: ' + str(maxindex))
    featvalue = sorted(featindex.items(), key=operator.itemgetter(1))

    fo = open(datapath + 'feat.bid.all.txt', 'w')
    fo.write(str(maxindex) + '\n')
    for fv in featvalue:
        fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
    fo.close()

    # indexing train
    print('indexing ' + datapath + 'train.bid.all.csv')
    fi = open(datapath + 'train.bid.all.csv', 'r')
    fo = open(datapath + 'train.bid.all.txt', 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        time_frac = s[4][8: 12]
        fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + to_time_frac(int(time_frac[0:2]), int(time_frac[2:4]), time_frac_dict))  # click + winning price + hour + timestamp
        index = featindex['truncate']
        fo.write(',' + str(index))
        for f in f1s:  # every direct first order feature
            col = namecol[f]
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        for f in f1sp:
            col = namecol[f]
            content = featTrans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        # col = namecol["usertag"]
        # tags = getTags(s[col])
        # # for tag in tags:
        # feat = str(col) + ':' + ''.join(tags)
        # if feat not in featindex:
        #     feat = str(col) + ':other'
        # index = featindex[feat]
        # fo.write(',' + str(index))
        fo.write('\n')
    fo.close()

    # indexing val
    print('indexing ' + datapath + 'val.bid.all.csv')
    fi = open(datapath + 'val.bid.all.csv', 'r')
    fo = open(datapath + 'val.bid.all.txt', 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        time_frac = s[4][8: 12]
        fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + to_time_frac(int(time_frac[0:2]), int(time_frac[2:4]),
                                                                      time_frac_dict))  # click + winning price + hour + timestamp
        index = featindex['truncate']
        fo.write(',' + str(index))
        for f in f1s:  # every direct first order feature
            col = namecol[f]
            if col >= len(s):
                print('col: ' + str(col))
                print(line)
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        for f in f1sp:
            col = namecol[f]
            content = featTrans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        # col = namecol["usertag"]
        # tags = getTags(s[col])
        # # for tag in tags:
        # feat = str(col) + ':' + ''.join(tags)
        # if feat not in featindex:
        #     feat = str(col) + ':other'
        # index = featindex[feat]
        # fo.write(',' + str(index))
        fo.write('\n')

    # indexing test
    print('indexing ' + datapath + 'test.bid.' + sample_type + '.csv')
    fi = open(datapath + 'test.bid.' + sample_type + '.csv', 'r')
    fo = open(datapath + 'test.bid.' + sample_type + '.txt', 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        time_frac = s[4][8: 12]
        fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + to_time_frac(int(time_frac[0:2]), int(time_frac[2:4]),
                                                                      time_frac_dict))  # click + winning price + hour + timestamp
        index = featindex['truncate']
        fo.write(',' + str(index))
        for f in f1s:  # every direct first order feature
            col = namecol[f]
            if col >= len(s):
                print('col: ' + str(col))
                print(line)
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        for f in f1sp:
            col = namecol[f]
            content = featTrans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        # col = namecol["usertag"]
        # tags = getTags(s[col])
        # # for tag in tags:
        # feat = str(col) + ':' + ''.join(tags)
        # if feat not in featindex:
        #     feat = str(col) + ':other'
        # index = featindex[feat]
        # fo.write(',' + str(index))
        fo.write('\n')
    fo.close()


def rand_sample(data_path):
    test_data = pd.read_csv(data_path + 'test.bid.all.csv')

    sample_indexs = sorted(random.sample(range(len(test_data)), int(len(test_data) * 0.7)))

    test_all_sample_data = test_data.iloc[sample_indexs, :]

    test_all_sample_data.to_csv(data_path + 'test.bid.rand.csv', index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi')
    parser.add_argument('--campaign_id', default='3427/', help='1458, 3358, 3386, 3427')
    parser.add_argument('--is_to_csv', default=True)

    setup_seed(1)

    args = parser.parse_args()

    data_path = args.data_path + args.dataset_name + args.campaign_id

    time_frac_dict = {}
    count = 0
    for i in range(24):
        hour_frac_dict = {}
        for item in [(0, 15), (15, 30), (30, 45), (45, 60)]:
            hour_frac_dict.setdefault(item, count)
            count += 1
        time_frac_dict.setdefault(i, hour_frac_dict)

    # if args.is_to_csv:
    #     print('to csv')
    #     day_indexs = pd.read_csv(data_path + 'day_indexs.csv', header=None).values.astype(int)
    #     train_indexs = day_indexs[day_indexs[:, 0] == 11][0]
    #     test_indexs = day_indexs[day_indexs[:, 0] == 12][0]
    #
    #     origin_train_data = pd.read_csv(data_path + 'train.all.origin.csv')
    #
    #     train_data = origin_train_data.iloc[:train_indexs[1], :] # 6-10
    #     val_data = origin_train_data.iloc[train_indexs[1]: train_indexs[2] + 1, :] # 11
    #     test_data = origin_train_data.iloc[train_indexs[2]:, :] # 12
    #
    #     train_data.to_csv(data_path + 'train.bid.all.csv', index=None)
    #     val_data.to_csv(data_path + 'val.bid.all.csv', index=None)
    #     test_data.to_csv(data_path + 'test.bid.all.csv', index=None)
    #
    # # no sample
    # to_libsvm_encode(data_path, 'all', time_frac_dict)

    # rand denotes random sample

    rand_sample(data_path)
    to_libsvm_encode(data_path, 'rand', time_frac_dict)




