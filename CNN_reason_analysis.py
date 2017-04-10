# -*- coding:utf-8 -*-
import os
import sys
import time
import json
import numpy
import random
import re
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
W_h = theano.shared(numpy.asarray(rng.uniform(low=-0.1,high=0.1,size=(200,300)),dtype=theano.config.floatX),borrow=True,name='W_h')
b_h = theano.shared(numpy.asarray(rng.uniform(low=-0.1,high=0.1,size=(300)),dtype=theano.config.floatX),borrow=True,name='b_h')
final_wc = W_c.eval()
final_bc = b_c.eval()
final_wh = W_h.eval()
final_bh = b_h.eval()
lr=0.0005


def ConvLayer(q1,q2):
    output = T.dot(T.concatenate([q1, q2]), W_c) + b_c
    return output

def load_embedding(path):
    word_embedding={}
    f=open(path,"r")
    c=f.readlines()
    f.close()
    for line in c:
        line = line.strip()
        word = line.split(' ')
        name = word[0].decode('utf-8')
        #print len(word)
        del word[0]
        word = [float(i) for i in word]
        if len(word)!=50:
            print len(word),word
        else:
            word = numpy.array(word)
            #print word[0],":",word[1:11]
            word_embedding[name] = word
    return word_embedding
print "loading embedding..."
word_embedding=load_embedding("vectors.bin")


    
def load_data(path):
    f = open(path,'r')
    data = pickle.load(f)
    f.close()
    label = []
    vec = []
    keys = []
    for key in data:

def load_dev_data(path):
    #将训练数据格式改成json
    f = open(path,"r")
    content = f.read()
    f.close()
    content = content.split('\n')
    jsonobj = []
    data = {}
    question = ""
    answer=[]
    not_answers=[]
    
    for line in content:
        line = line.split('\t')
        #新问题出现时
        if(line[0]!=question and question != ""):
            data["question"]=question
            data["answer"]=answer
            data["not_answers"]=not_answers
            if(path == "nlpcc-dev-data"):
                type = answer_type[question]
                score = 0.06
                res = []
                if(type == "TIM"):
                    for ans in answer:
                        words = segmentor.segment(ans)
                        postag = postagger.postag(words)
                        flag=0
                        for each_pos in postag:
                            if each_pos == "nt":
                                flag=1
                                res.append(score)
                                break
                        if(flag==0):
                            res.append(0)
                    for ans in not_answers:
                        words = segmentor.segment(ans)
                        postag = postagger.postag(words)
                        flag=0
                        for each_pos in postag:
                            if each_pos == "nt":
                                flag=1
                                res.append(score)
                                break
                        if(flag==0):
                            res.append(0)
                elif(type == "LOC"):
                    for ans in answer:
                        words = segmentor.segment(ans)
                        postag = postagger.postag(words)
                        netags = recognizer.recognize(words, postag)
                        flag=0
                        for each_tag in netags:
                            if each_tag[2:]=="Ns":
                                flag=1
                                res.append(score)
                                break
                        if(flag==0):
                            res.append(0)
                    for ans in not_answers:
                        words = segmentor.segment(ans)
                        postag = postagger.postag(words)
                        netags = recognizer.recognize(words, postag)
                        flag=0
                        for each_tag in netags:
                            if each_tag[2:]=="Ns":
                                flag=1
                                res.append(score)
                                break
                        if(flag==0):
                            res.append(0)
                elif(type == "HUM"):
                    for ans in answer:
                        words = segmentor.segment(ans)
                        postag = postagger.postag(words)
                        netags = recognizer.recognize(words, postag)
                        flag=0
                        for each_tag in netags:
                            if each_tag[2:]=="Nh":
                                flag=1
                                res.append(score)
                                break
                        if(flag==0):
                            res.append(0)
                    for ans in not_answers:
                        words = segmentor.segment(ans)
                        postag = postagger.postag(words)
                        netags = recognizer.recognize(words, postag)
                        flag=0
                        for each_tag in netags:
                            if each_tag[2:]=="Nh":
                                flag=1
                                res.append(score)
                                break
                        if(flag==0):
                            res.append(0)
                elif(type == "NUM"):
                    for ans in answer:
                        words = segmentor.segment(ans)
                        postag = postagger.postag(words)
                        flag=0
                        for each_pos in postag:
                            if each_pos == "m":
                                flag=1
                                res.append(score)
                                break
                        if(flag==0):
                            res.append(0)
                    for ans in not_answers:
                        words = segmentor.segment(ans)
                        postag = postagger.postag(words)
                        flag=0
                        for each_pos in postag:
                            if each_pos == "m":
                                flag=1
                                res.append(score)
                                break
                        if(flag==0):
                            res.append(0)
                else:
                    for ans in answer:
                        res.append(0)
                    for ans in not_answers:
                        res.append(0)
                dev_type.append(res)
            
            jsonobj.append(data)
            answer = []
            not_answers = []
            data = {}
        
        question = line[0]
        ans = line[1]
        label = line[2]
        if(label == "1"):
            answer.append(ans)
        else:
            not_answers.append(ans)
    #处理最后一组数据
    data["question"]=question
    data["answer"]=answer
    data["not_answers"]=not_answers
    jsonobj.append(data)
    if(path == "nlpcc-dev-data"):
        res = []
        for ans in answer:
            res.append(0)
        for ans in not_answers:
            res.append(0)
        dev_type.append(res)
    #检验数据处理对否正确
    """
    print len(jsonobj)
    cnt=0
    for case in jsonobj:
        #print case["question"]
        cnt += len(case["answer"])
    print cnt
    """
    #test_data = jsonobj
    #分词，将中文转换成embedding
    vector_corpus = []
    apara_cnt = 0
    ahyper_cnt = 0
    npara_cnt = 0
    nhp_cnt=0
    hp_cnt=0
    nhyper_cnt = 0
    for case in jsonobj:
        vector = {}
        #question
        #分词时候输入的utf-8编码
        q_words = segmentor.segment(case['question'])
        q_vec = []
        
        for word in q_words:
            if(word in stop_words):
                continue
            try :
                #字典中key是unicode编码
                q_vec.append(word_embedding[word.decode('utf-8')])
            except KeyError:
                temp = rng.uniform(low=-0.1,high=0.1,size=[50])
                q_vec.append(temp)
        
        #句子长度为1，添加一个词
        while(len(q_vec)<=1):
            temp = rng.uniform(low=-0.1,high=0.1,size=[50])
            q_vec.append(temp)
        vector['question'] = q_vec
        
        #answer
        answer = []
        for stc in case['answer']:
            a_words = segmentor.segment(stc)
            a_vec = []
            
            for word in a_words:
                #print word
                flag=0
                
                if(word in stop_words):
                    continue
                if(word in paraphrase and word not in q_words):
                    for para_word in paraphrase[word]:
                        if(para_word in q_words):
                            apara_cnt += 1
                            flag=1
                            try :
                                a_vec.append(word_embedding[para_word.decode('utf-8')])
                                #print "para word:",para_word.decode("utf-8")
                            except KeyError:
                                temp = rng.uniform(low=-0.1,high=0.1,size=[50])
                                a_vec.append(temp)
                            break
                if(flag==0 and word not in q_words and len(word)>3):
                    try:
                        cursor.execute("select name from check_hyper where id in (select hyper_id from check_hhpair where entity_id = (SELECT id FROM check_entity WHERE name = '"+ word.decode('utf-8') +"') and (score>=0.985 or correct_count >0))")
                        #print a_words[i].decode('utf-8')
                        values = cursor.fetchall()
                        for j in range(len(values)):
                            if values[j][0].encode("utf-8") != "人" and values[j][0].encode("utf-8") in q_words:
                                ahyper_cnt += 1
                                flag=1
                                try :
                                    a_vec.append(word_embedding[values[j][0]])
                                    #print "hyper word:",values[j][0]
                                except KeyError:
                                    temp = rng.uniform(low=-0.1,high=0.1,size=[50])
                                    a_vec.append(temp)
                            
                    except:
                        print "can't search database:",word.decode('utf-8')
                
                if(flag==0):
                    try :
                        a_vec.append(word_embedding[word.decode('utf-8')])
                    except KeyError:
                        temp = rng.uniform(low=-0.1,high=0.1,size=[50])
                        a_vec.append(temp)
            #句子长度为1，添加一个词
            while(len(a_vec)<=1):
                temp = rng.uniform(low=-0.1,high=0.1,size=[50])
                a_vec.append(temp)
            answer.append(a_vec)
        vector['answer'] = answer
        
        #not_answers
        not_answers = []
        for stc in case['not_answers']:
            a_words = segmentor.segment(stc)
            a_vec = []
            for word in a_words:
                #print word
                if(word in stop_words):
                    continue
                flag=0
                if(word in paraphrase and word not in q_words):
                    for para_word in paraphrase[word]:
                        if(para_word in q_words):
                            npara_cnt += 1
                            flag=1
                            try :
                                a_vec.append(word_embedding[para_word.decode('utf-8')])
                                #print "para word:",para_word.decode("utf-8")
                            except KeyError:
                                temp = rng.uniform(low=-0.1,high=0.1,size=[50])
                                a_vec.append(temp)
                            break
                
                if(flag==0 and word not in q_words and len(word)>3):
                    try:
                        cursor.execute("select name from check_hyper where id in (select hyper_id from check_hhpair where entity_id = (SELECT id FROM check_entity WHERE name = '"+ word +"') and (score>=0.985 or correct_count >0))")
                        #print a_words[i].decode('utf-8')
                        values = cursor.fetchall()
                        #print len(values)
                        for j in range(len(values)):
                            #print values[j][0]
                            if values[j][0].encode("utf-8") != "人" and values[j][0].encode("utf-8") in q_words:
                                nhyper_cnt += 1
                                flag=1
                                try :
                                    a_vec.append(word_embedding[values[j][0]])
                                    #print "hyper word:",values[j][0]
                                except KeyError:
                                    temp = rng.uniform(low=-0.1,high=0.1,size=[50])
                                    a_vec.append(temp)
                    except:
                        print "can't search database:",word.decode('utf-8')
                
                if(flag==0):
                    try :
                        a_vec.append(word_embedding[word.decode('utf-8')])
                    except KeyError:
                        temp = rng.uniform(low=-0.1,high=0.1,size=[50])
                        a_vec.append(temp)
            #句子长度为1，添加一个词
            while(len(a_vec)<=1):
                temp = rng.uniform(low=-0.1,high=0.1,size=[50])
                a_vec.append(temp)
            not_answers.append(a_vec)
        vector['not_answers'] = not_answers

        vector_corpus.append(vector)
        
    #print len(vector_corpus)
    print "answer para:",apara_cnt
    print "answer hyper:",ahyper_cnt
    
    print "not answer para:",npara_cnt
    print "not answer hyper:",nhyper_cnt
    return vector_corpus,jsonobj
    
def wmpredict(stc,ans_stc):
    cnt = 0.0
    #nt process
    q_words = segmentor.segment(stc)
    q_postag = postagger.postag(q_words)
    answer = segmentor.segment(ans_stc)
    i=-1
    flag=0
    for pos in q_postag:
        i+=1
        res = re.findall('[0-9]+',q_words[i])
        if "什么" in q_words and pos =="nt" and len(res)>0:
            for a_word in answer:
                if(a_word == q_words[i]):
                    #print "nt word:",q_words[i]
                    cnt+=10
                    flag+=1
            break
    
    #focus
    if(len(focus[stc])>2):
        if focus[stc][-1]!="时候" and focus[stc][-1] in answer:
            cnt += 10
            flag+=1
            #print stc.decode('utf-8'),focus[stc][-1].decode('utf-8')
    for a_word in answer:
        if(a_word in q_words):
            cnt+=1
    return cnt,flag
    
def CNN_train():
    #训练模型
    #y = T.lscalar()  
    x1 = T.matrix('x1')
    x2 = T.matrix('x2')
    q = T.matrix('q')
    #y = T.ivector('y')
    
    #convolution
    conv_question_output, _ = theano.scan(fn=ConvLayer, sequences=[q[:-1],q[1:]])
    conv_answer_output, _ = theano.scan(fn=ConvLayer, sequences=[x1[:-1],x1[1:]])
    conv_notanswer_output, _ = theano.scan(fn=ConvLayer, sequences=[x2[:-1],x2[1:]])
    
    #tanh and pooling
    q_hidden = T.tanh(T.max(conv_question_output,axis=0))
    a_hidden = T.tanh(T.max(conv_answer_output,axis=0))
    n_hidden = T.tanh(T.max(conv_notanswer_output,axis=0))
    
    #hidden layer
    q_output = T.tanh(T.dot(q_hidden,W_h)+b_h)
    a_output = T.tanh(T.dot(a_hidden,W_h)+b_h)
    n_output = T.tanh(T.dot(n_hidden,W_h)+b_h)
    
    #cosine similarity
    cos_sim1 = T.sum(q_output*a_output)/T.sqrt(T.sum(q_output**2) * T.sum(a_output**2))
    #out1 = T.nnet.sigmoid(cos_sim1)
    
    cos_sim2 = T.sum(q_output*n_output)/T.sqrt(T.sum(q_output**2) * T.sum(n_output**2))
    #out2 = T.nnet.sigmoid(cos_sim2)
    
    cost = T.max([0,(0.5-cos_sim1+cos_sim2)])
    
    gparams = []
    params = [W_c,b_c,W_h,b_h]
    for param in params:
        gparams.append(T.grad(cost,param))
        
    updates={}
    for param,gparam in zip(params,gparams):
        upd = param - lr*gparam
        updates[param] = upd
    train_model = theano.function(inputs=[q,x1,x2],outputs=[cost,cos_sim1,cos_sim2],updates=updates)
    
    #预测模型
    px1 = T.matrix('px1')
    px2 = T.matrix('px2')
        
    #convolution
    conv_question_output, _ = theano.scan(fn=ConvLayer, sequences=[px1[:-1],px1[1:]])
    conv_answer_output, _ = theano.scan(fn=ConvLayer, sequences=[px2[:-1],px2[1:]])
        
    #tanh and pooling
    q_hidden = T.tanh(T.max(conv_question_output,axis=0))
    a_hidden = T.tanh(T.max(conv_answer_output,axis=0))

    #hidden layer
    q_output = T.tanh(T.dot(q_hidden,W_h)+b_h)
    a_output = T.tanh(T.dot(a_hidden,W_h)+b_h)
        
    cos_sim = T.sum(q_output*a_output)/T.sqrt(T.sum(q_output**2) * T.sum(a_output**2))
    out = T.nnet.sigmoid(cos_sim)
        
    predict = theano.function(inputs=[px1,px2],outputs=out)
    
    #开始训练
    print "begin training..."
    iter = 20
    maxmrr = 0.0
    while(iter):
        cnt=0
        all_cost=0
        print "iter:",iter
        for case in train_data:
            for i in range(len(case['answer'])):
                for j in range(len(case['not_answers'])):
                    result = train_model(case['question'],case['answer'][i],case['not_answers'][j])
                    all_cost += result[0]
        print "cost:",all_cost
        
        mrr=0.0
        map=0.0
        k=0
        for case in dev_data:
            sim = {}
            word_sim = {}
            flag={}
            ans_cnt=len(case['answer'])
            #if(ans_cnt==0):
                #print "no answer id :",k
            i=1
            maxcnt = 0.0
            if(len(case['answer']) + len(case['not_answers']) != len(dev_type[k])):
                print k
                print "stc:",len(case['answer']) + len(case['not_answers'])
                print "score:",len(dev_type[k])
                print "answer type wrong!!"
            for stc in case["answer"]:
                sim[i] = predict(case['question'],stc)
                sim[i] += dev_type[k][i-1]
                word_sim[i],flag[i] = wmpredict(raw_data[k]['question'],raw_data[k]['answer'][i-1])
                if(word_sim[i]>maxcnt):
                    maxcnt = word_sim[i]
                i+=1
            for stc in case["not_answers"]:
                sim[i] = predict(case['question'],stc)
                sim[i] += dev_type[k][i-1]
                word_sim[i],flag[i] = wmpredict(raw_data[k]['question'],raw_data[k]['not_answers'][i-1-ans_cnt])
                if(word_sim[i]>maxcnt):
                    maxcnt = word_sim[i]
                i+=1
            #if(maxcnt == 0):
            #    print raw_data[k]['question']
            for j in range(1,i):
                if flag[j] > 0:
                    sim[j] += flag[j]
                else:
                    sim[j] += word_sim[j]/(1+maxcnt*15)
            res = sorted(sim.iteritems(), key=lambda d: d[1],reverse = True)
            #print res[0][1],res[1][1]
            for i in range(len(res)):
                if(res[i][0]<=ans_cnt):
                    mrr += 1.0/(i+1)
                    break
            cnt=1.0
            pk = 0.0
            for i in range(len(res)):
                if(res[i][0]<=ans_cnt):
                    pk += cnt/(i+1)
                    cnt += 1.0
            if(ans_cnt!=0):
                pk = pk/ans_cnt
            map += pk
            k+=1
        mrr = mrr/len(dev_data)
        map = map/len(dev_data)
        print "dev_data"
        print "MRR: ",mrr
        print "MAP: ",map
        
        if(mrr > maxmrr and mrr > 0.73):
            maxmrr = mrr
            f = open("submitresults.txt","w")
            f2 = open("submit.txt","w")
            k=0
            for case in test_data:
                sim = {}
                word_sim = {}
                flag={}
                ans_cnt=len(case['answer'])
                #if(ans_cnt==0):
                    #print "no answer id :",k
                i=1
                maxcnt=0.0
                for stc in case["answer"]:
                    sim[i] = predict(case['question'],stc)
                    sim[i] += test_type[k][i-1]
                    word_sim[i],flag[i] = wmpredict(test_raw_data[k]['question'],test_raw_data[k]['answer'][i-1])
                    if(word_sim[i]>maxcnt):
                        maxcnt = word_sim[i]
                    i+=1
                    
                for j in range(1,i):
                    if flag[j] > 0:
                        sim[j] += flag[j]
                    else:
                        sim[j] += word_sim[j]/(1+maxcnt*15)
                for i in range(len(sim)):
                    f2.write(str(sim[i+1])+"\n")
                res = sorted(sim.iteritems(), key=lambda d: d[1],reverse = True)
                
                
                f.write(test_raw_data[k]['question']+"\n")
                for i in range(len(res)):
                    #print res[i][0]
                    f.write(test_raw_data[k]['answer'][res[i][0]-1]+"\n")
                f.write("-----------------\n")
                k+=1
            f.close()
            f2.close()
        iter -= 1
    

        
if __name__ == '__main__':
    print "loading training data..."
    train_data,_ = load_dev_data("nlpcc-train-data")
    print "loading dev data..."
    dev_data,raw_data = load_dev_data("nlpcc-dev-data")
    print "loading test data..."
    test_data,test_raw_data = load_test_data("evatestdata2-dbqa.testing-data")
    #print "train_data length:",len(train_data)
    word_embedding.clear()
    

    CNN_train()
    #load_data("nlpcc-iccpol-2016.dbqa.training-data")
    #CNN_predict()
    
    
