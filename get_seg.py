#encoding:utf-8
import os
import json
import getopt
import time
import sys
import random
import copy
import math
import pickle
import pyltp

MODELDIR = "/home/llyu/work/ltpdata/3.3.1/ltp_data"
from pyltp import Segmentor
segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))

'''
功能：处理单个文件，获取id、text、label
'''
def get_one_data(path,lib_path='/home/llyu/data_work/lib/useless.stc'):
    # get useless sentence
    useless_stc = []
    f = open(lib_path,'r')
    for line in f:
        if line.strip()!='':
            useless_stc.append(line.strip())
    f.close()

    f = open(path,'r')
    data = {}
    cur_id = ''
    cur_label = ''
    text = ''
    for line in f:
        line = line.strip().split('|')
        if len(line) != 2:
            pass
        # text line
        #elif line[0]=='0' or line[0]=='1':
        elif line[0]=='1':
            # skip useless sentence
            flag = 0
            for stc in useless_stc:
                if line[1].find(stc)!=-1:
                    flag = 1
            if flag == 0 :#and cur_cnt<5:
                text += line[1]
        # id line
        elif len(line[0])>10:
            if cur_id != '':
                data[cur_id]=(cur_label,text)
            cur_id = line[0]
            cur_label = line[1]
            text = ''
    f.close()
    if cur_id!='':
        data[cur_id]=(cur_label,text)
    return data


'''
功能：读取所有数据
'''
def write_all_data(path,output_path):
    if os.path.isfile(path):
        files=[path]
    else:
        files = os.listdir(path)
        for i in range(len(files)):
            files[i] = path+'/'+files[i]
    alldata = {}
    f = open(output_path,'w')
    for each in files:
        data = get_one_data(each)
        for key in data:
            words = segmentor.segment(data[key][1])
            label = data[key][0]
            text = ' '.join(words)
            f.write(text+'\n')
            if key not in alldata:
                alldata[key]=(label,text)
    f.close()
    return alldata


# 划分训练集和测试集
def divide_corpus(data):
    test_keys = random.sample(data.keys(),len(data)/5)
    test = {}
    train = {}
    for key in data:
        label = data[key][0]
        try:
            a = int(label)
        except:
            continue
        text = data[key][1]
        if key in test_keys:
            test[key]=(label,text)
        else:
            train[key]=(label,text)
    f = open('lib/train.pickle','w')
    pickle.dump(train,f)
    f.close()
    f = open('lib/test.pickle','w')
    pickle.dump(test,f)
    f.close()
    return



'''
-i:输入路径
-o:输出路径
功能：分词；类别均衡；获取词典写入文件；向量转换写入文件；
'''
if __name__ == '__main__':
    input_path = ''
    output_path = 'default_name'
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:o:', [])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o, a in opts:
        if o in ('-i'):
            input_path = a
        elif o in ('-o'):
            output_path = a
    if input_path != '' and os.path.exists(input_path):
        data = write_all_data(input_path,output_path)
        divide_corpus(data)

