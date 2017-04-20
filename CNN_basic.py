# -*- coding:utf-8 -*-
import os
import re
import sys
import time
import json
import numpy
import random
import pickle
import theano
import theano.tensor as T
MODELDIR="/home/llyu/work/ltpdata/3.3.1/ltp_data"
from pyltp import Segmentor

segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))

rng = numpy.random.RandomState(23455)
#convolution parameter
W_c = theano.shared(numpy.asarray(rng.uniform(low=-0.1,high=0.1,size=(100,200)),dtype=theano.config.floatX),borrow=True,name='W_c')
b_c = theano.shared(numpy.asarray(rng.uniform(low=-0.1,high=0.1,size=(200)),dtype=theano.config.floatX),borrow=True,name='b_c')
W_h = theano.shared(numpy.asarray(rng.uniform(low=-0.1,high=0.1,size=(200,4)),dtype=theano.config.floatX),borrow=True,name='W_h')
b_h = theano.shared(numpy.asarray(rng.uniform(low=-0.1,high=0.1,size=(4)),dtype=theano.config.floatX),borrow=True,name='b_h')
final_wc = W_c.eval()
final_bc = b_c.eval()
final_wh = W_h.eval()
final_bh = b_h.eval()
lr=0.005


def ConvLayer(q1,q2):
    output = T.dot(T.concatenate([q1, q2]), W_c) + b_c
    return output

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
        if len(word)!=50:
            print 'word embedding wrong',len(word),word
        else:
            word = numpy.array(word)
            #print word[0],":",word[1:11]
            word_embedding[name] = word
    f.close()
    return word_embedding

def load_raw_text(path):
    f = open(path,'r')
    data = pickle.load(f)
    f.close()
    return data
    
def load_data(path):
    f = open(path,'r')
    data = pickle.load(f)
    f.close()
    label = []
    vec = []
    keys = []
    for key in data:
        cur_label = data[key][0]
        if data[key][0]=='1':
            cur_label = numpy.array([1,0,0,0])
        elif data[key][0]=='2':
            cur_label = numpy.array([0,1,0,0])
        elif data[key][0]=='3':
            cur_label = numpy.array([0,0,1,0])
        elif data[key][0]=='4':
            cur_label = numpy.array([0,0,0,1])
        else:
            continue
            #print 'label goes wrong',key,data[key][0]
        text = data[key][1].decode('utf-8').strip()
        text = text.split(' ')
        cur_vec = []
        for word in text:
            if word in word_embedding:
                cur_vec.append(word_embedding[word])
            else:
                temp = rng.uniform(low=-0.1,high=0.1,size=[50])
                cur_vec.append(temp)
        while len(cur_vec)<=1:
            cur_vec.append(rng.uniform(low=-0.1,high=0.1,size=[50]))
        keys.append(key)
        vec.append(numpy.array(cur_vec))
        label.append(cur_label)
    # vec.shape: instance,words,embedding
    return keys,numpy.array(vec),numpy.array(label)


    
    
def CNN_train():
    #训练模型
    #y = T.lscalar()  
    x = T.matrix('x')
    y = T.vector('y')
    
    #convolution
    conv_x_output, _ = theano.scan(fn=ConvLayer, sequences=[x[:-1],x[1:]])
    
    #tanh and pooling
    x_hidden = T.tanh(T.max(conv_x_output,axis=0))
    
    #hidden layer
    x_output = T.tanh(T.dot(x_hidden,W_h)+b_h)
    
    #softmax layer
    softmax_output = T.nnet.softmax(x_output)
    
    cost = -T.log(T.sum(y*softmax_output))
    #sigmoid layer
    #sigmoid_output = T.nnet.sigmoid(x_output)
    #cost = T.sum(-y*T.log(sigmoid_output) - (1-y)*T.log(1-sigmoid_output))
    
    gparams = []
    params = [W_c,b_c,W_h,b_h]
    for param in params:
        gparams.append(T.grad(cost,param))
        
    updates={}
    for param,gparam in zip(params,gparams):
        upd = param - lr*gparam
        updates[param] = upd
    train_model = theano.function(inputs=[x,y],outputs=[cost,softmax_output],updates=updates)
    
    #预测模型
    px = T.matrix('px')
    #convolution
    conv_px_output, _ = theano.scan(fn=ConvLayer, sequences=[px[:-1],px[1:]])
    #tanh and pooling
    px_hidden = T.tanh(T.max(conv_px_output,axis=0))
    #hidden layer
    px_output = T.tanh(T.dot(px_hidden,W_h)+b_h)
    #output layer 
    px_softmax_output = T.nnet.softmax(px_output)


    predict = theano.function(inputs=[px],outputs=px_softmax_output)
    
    #开始训练
    print "begin training..."
    iter = 50
    maxacc = 0.0
    while(iter):
        cnt=0
        all_cost=0
        print "iter:",iter
        for i in range(len(train_vec)):
            #print train_vec[i].shape,train_label[i].shape
            result = train_model(train_vec[i],train_label[i])
            all_cost += result[0]
        print "cost:",all_cost
        
        cnt = 0
        for i in range(len(train_vec)):
            result = predict(train_vec[i])
            if train_label[i].argmax() == result.argmax():
                cnt += 1
        print 'train acc:',float(cnt)/len(train_vec)

        cnt = 0
        for i in range(len(dev_vec)):
            result = predict(dev_vec[i])
            if dev_label[i].argmax() == result.argmax():
                cnt += 1
        print 'dev acc:',float(cnt)/len(dev_vec)
        print '-------------'
        iter -= 1

        if float(cnt)/len(dev_vec) < maxacc:
            continue
        # 记录参数，写入文件
        param = {}
        param['wc'] = W_c.eval()
        param['bc'] = b_c.eval()
        param['wh'] = W_h.eval()
        param['bh'] = b_h.eval()
        f = open('lib/parameter.pickle','w')
        pickle.dump(param,f)
        f.close()
        
        # 将正确及错误用例分别写入文件
        maxacc = float(cnt)/len(dev_vec)
        f = open('cnn_wrong','w')
        rf = open('cnn_right','w')
        for i in range(len(dev_vec)):
            result = predict(dev_vec[i])
            if dev_label[i].argmax() != result.argmax():
                f.write(dev_keys[i]+'|'+str(result.argmax()+1)+'|'+str(dev_label[i].argmax()+1)+'|'+str(result)+'\n')
                f.write(raw_text[dev_keys[i]]+'\n')
            else:
                rf.write(dev_keys[i]+'|'+str(result.argmax()+1)+'|'+str(dev_label[i].argmax()+1)+'|'+str(result)+'\n')
                rf.write(raw_text[dev_keys[i]]+'\n')

        f.close()
        rf.close()
    return


if __name__ == '__main__':
    print "loading embedding..."
    word_embedding=load_embedding("vectors.bin")
    print "loading training data..."
    train_keys,train_vec,train_label = load_data("lib/train.pickle")
    print "loading dev data..."
    dev_keys,dev_vec,dev_label = load_data("lib/test.pickle")
    word_embedding.clear()
    
    raw_text = load_raw_text('lib/raw_text.pickle')

    CNN_train()
    
    
