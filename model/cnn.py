# -*- coding:utf-8 -*-
import os
import sys
import numpy
import pickle
import chardet
import theano
import theano.tensor as T
from pyltp import Segmentor

def load_conf(path='cnn.conf'):
    seg_path = ''
    para_path = ''
    embed_path = ''
    f = open(path,'r')
    for line in f:
        line = line.strip().split('=')
        if len(line)==2:
            if line[0].strip()=='segment_path':
                seg_path = line[1].strip()
            elif line[0].strip()=='parameter_path':
                para_path = line[1].strip()
            elif line[0].strip()=='embedding_path':
                embed_path = line[1].strip()
    f.close()
    if seg_path=='' or para_path=='' or embed_path=='':
        print 'cnn.conf went wrong...'
        sys.exit(2)
    return seg_path,para_path,embed_path
print 'loading conf...'
MODELDIR,para_path,embedding_path=load_conf()

segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))
rng = numpy.random.RandomState(23455)

def load_parameter(path):
    f = open(path,'r')
    param = pickle.load(f)
    f.close()

    #convolution parameter
    W_c = theano.shared(param['wc'],borrow=True,name='W_c')
    b_c = theano.shared(param['bc'],borrow=True,name='b_c')
    W_h = theano.shared(param['wh'],borrow=True,name='W_h')
    b_h = theano.shared(param['bh'],borrow=True,name='b_h')
    return W_c,b_c,W_h,b_h
print 'loading parameter...'
W_c,b_c,W_h,b_h = load_parameter(para_path)


def load_embedding(path):
    word_embedding={}
    f=open(path,"r")
    for line in f:
        line = line.strip()
        word = line.split(' ')
        name = word[0].decode('utf-8')
        #print len(word)
        del word[0]
        word = [float(i) for i in word]
        if len(word)==50:
            word = numpy.array(word)
            word_embedding[name] = word
    f.close()
    return word_embedding
print "loading embedding..."
word_embedding=load_embedding(embedding_path)



def trans2utf(text):
    if type(text)==unicode:
        return text.encode('utf-8')
    elif type(text)==str:
        codetype = chardet.detect(text)['encoding']
        if codetype == 'utf-8':
            return text
        elif codetype == 'GB2312' or codetype=='GBK' or codetype=='GB18030' or codetype=='TIS-620':
            return text.decode('gb18030').encode('utf-8')
        else:
            try:
                text = text.encode('utf-8')
                return text
            except:
                print 'type unknown',codetype
                sys.exit(2)
    else:
        print 'type unknown',type(text)
        sys.exit(2)


def trans2words(text):
    need = []
    text = text.strip().split('\n')
    for line in text:
        line = line.strip().split('\001\002\003')
        if len(line)==2 and line[0]=='1':
            words = line[1].strip().split(' ')
            for word in words:
                need.append(word.decode('utf-8'))
    return need


def ConvLayer(q1,q2):
    output = T.dot(T.concatenate([q1, q2]), W_c) + b_c
    return output


def find_nd(score):
    nd = 0.0
    pos = -1
    for i in range(len(score)):
        if score[i]>nd and score[i]<max(score):
            nd = score[i]
            pos = i
    return pos+1,nd


'''
输入：text
输出：label
'''
def text_predict(text):
    # 文本转向量
    text = trans2utf(text)
    words = segmentor.segment(text)
    cur_vec = []
    for word in words:
        word = word.decode('utf-8')
        if word in word_embedding:
            cur_vec.append(word_embedding[word])
        else:
            temp = rng.uniform(low=-0.1,high=0.1,size=[50])
            cur_vec.append(temp)

    # rule 1: 词数过少，判定为4
    if len(cur_vec)<2:
        return 4

    # 预测模型
    px = T.matrix('px')
    # convolution
    conv_px_output, _ = theano.scan(fn=ConvLayer, sequences=[px[:-1],px[1:]])
    # tanh and pooling
    px_hidden = T.tanh(T.max(conv_px_output,axis=0))
    # hidden layer
    px_output = T.tanh(T.dot(px_hidden,W_h)+b_h)
    # output layer 
    px_softmax_output = T.nnet.softmax(px_output)
    # function
    model_predict = theano.function(inputs=[px],outputs=px_softmax_output)
    
    result = model_predict(numpy.array(cur_vec))[0]
    # rule 2: 最高分过低，默认为3类
    if max(result)<0.41:
        return 3,result
    # rule 3: 识别为3类，但3类分数与次高分数相差很小，选用次高的类别
    if result.argmax()==2:
        pos,nd = find_nd(result)
        if max(result)-nd<0.01:
            return pos,result
    return result.argmax()+1,result



'''
输入：segmented text
输出：label
'''
def predict(text):
    # 文本转向量
    text = trans2utf(text)
    words = trans2words(text)
    cur_vec = []
    cnt = 0
    for word in words:
        if word in word_embedding:
            cur_vec.append(word_embedding[word])
            cnt += 1
        else:
            temp = rng.uniform(low=-0.1,high=0.1,size=[50])
            cur_vec.append(temp)
    # print 'vector length:',len(cur_vec)
    # rule 1: 词数过少，判定为4
    if cnt < 2:
        return 4

    # 预测模型
    px = T.matrix('px')
    # convolution
    conv_px_output, _ = theano.scan(fn=ConvLayer, sequences=[px[:-1],px[1:]])
    # tanh and pooling
    px_hidden = T.tanh(T.max(conv_px_output,axis=0))
    # hidden layer
    px_output = T.tanh(T.dot(px_hidden,W_h)+b_h)
    # output layer 
    px_softmax_output = T.nnet.softmax(px_output)
    # function
    model_predict = theano.function(inputs=[px],outputs=px_softmax_output)
    
    result = model_predict(numpy.array(cur_vec))[0]
    # rule 2: 最高分过低，默认为3类
    if max(result)<0.41:
        return 3
    # rule 3: 识别为3类，但3类分数与次高分数相差很小，选用次高的类别
    if result.argmax()==2:
        pos,nd = find_nd(result)
        if max(result)-nd<0.01:
            return pos
    return result.argmax()+1

if __name__ == '__main__':
   #print predict('1\001\002\003移动 怎么 乱 扣费 啊')
   while(True):
       order = raw_input('请输入文本:')
       if order == 'exit':
           break
       print text_predict(order)
    
