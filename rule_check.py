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
import numpy

MODELDIR = "/home/llyu/work/ltpdata/3.3.1/ltp_data"
from pyltp import Segmentor
segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))


def load_embedding(path):
    word_embedding={}
    f= open(path,'r')
    for line in f:
        line = line.strip()
        word = line.split(' ')
        name = word[0].decode('utf-8')
        del word[0]
        word = [float(i) for i in word]
        if len(word)!=50:
            print 'word embedding wrong',len(word),word
        else:
            word = numpy.array(word)
            word_embedding[name]=word
    f.close()
    return word_embedding




'''
功能：处理单个文件，获取id、text、label
'''
def get_one_data(path):
    # get useless sentence

    f = open(path,'r')
    keys = []
    label = []
    predict = []
    score = []
    for line in f:
        line = line.strip().split('|')
        if len(line) == 4:
            keys.append(line[0])
            predict.append(line[1])
            label.append(line[2])
            cur = line[3].strip().replace(']','').split(' ')
            cur_score = []
            for one in cur:
                if len(one)>5:
                    cur_score.append(float(one))
            score.append(cur_score)
    f.close()
    return keys,predict,label,score


def find_nd(score):
    nd = 0.0
    pos = -1
    for i in range(len(score)):
        if score[i]>nd and score[i]<max(score):
            nd = score[i]
            pos = i
    return pos+1,nd


def check_rule_4(path):
    f = open(path,'r')
    data = pickle.load(f)
    f.close()
    
    all_cnt = 0
    right_cnt = 0
    rkeys = []
    wkeys = []
    for key in data:
        try:
            label = int(data[key][0])
        except:
            print 'label not int',key,':',data[key][0]
            continue
        text = data[key][1].decode('utf-8').strip()
        text = text.split(' ')
        word_cnt=0
        for word in text:
            if word in word_embedding:
                word_cnt += 1
        #if word_cnt<2:
        if len(text)<2:
            all_cnt += 1
            if label == 4:
                right_cnt += 1
                rkeys.append(key)
            else:
                wkeys.append(key)
    print right_cnt,all_cnt,float(right_cnt)/all_cnt
    return rkeys,wkeys


'''
-i:输入路径
-o:输出路径
'''
if __name__ == '__main__':
    input_path = ['cnn_right_softmax','cnn_wrong_softmax']
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
    right_keys,right_predict,right_label,right_score = get_one_data(input_path[0])
    wrong_keys,wrong_predict,wrong_label,wrong_score = get_one_data(input_path[1])
    '''
    # 大于0.7的93% 分类正确
    cnt = 0
    for i in range(len(right_keys)):
        if max(right_score[i])>=0.7:
            cnt += 1
    print 'right:',len(right_keys),cnt,float(cnt)/len(right_keys)
    cnt = 0
    for i in range(len(wrong_keys)):
        if max(wrong_score[i])>=0.7:
            cnt += 1
    print 'wrong:',len(wrong_keys),cnt,float(cnt)/len(wrong_keys)
    '''

    '''
    # 最高分值低于0.41 默认识别为3类
    for threshold in range(41,42):
        threshold /=100.0
        print threshold
        cnt = 0
        for i in range(len(right_keys)):
            nd_pos,nd_score = find_nd(right_score[i])
            #if int(right_predict[i])==3 and max(right_score[i])-nd_score<threshold:
            if max(right_score[i])<threshold and int(right_label[i])!=3:
                cnt -= 1
        print cnt
        for i in range(len(wrong_keys)):
            nd_pos,nd_score = find_nd(wrong_score[i])
            #if int(wrong_predict[i])==3 and max(wrong_score[i])-nd_score<threshold and nd_pos==int(wrong_label[i]) :
            if max(wrong_score[i])<threshold and int(wrong_label[i])==3:
            #if nd_pos==int(wrong_label[i]):
                cnt += 1
        print cnt#,len(wrong_label),float(cnt)/len(wrong_label)
    '''

    # 词数过少（1个词），判定为4， 234/284
    print "loading embedding..."
    word_embedding=load_embedding("vectors.bin")
    rkeys,wkeys = check_rule_4('lib/test.pickle')
    for threshold in range(1,2):
        w2r_cnt = 0
        r2w_cnt = 0
        for key in rkeys:
            if key in wrong_keys:
                w2r_cnt += 1
        for key in wkeys:
            if key in right_keys:
                r2w_cnt += 1
        print 'w2r:',len(rkeys),w2r_cnt
        print 'r2w:',len(wkeys),r2w_cnt

