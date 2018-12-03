#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 20:46:27 2018

@author: xuweijia
"""
import torch.nn as nn
import torch
from torch.autograd import Variable
from collections import Counter, OrderedDict
stop_words=[]
with open('stopword.txt','r') as f:
    for line in f:
        stop_words.append(line.strip())
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',"''",'``',"'s","-","--",'–']
stop_words.extend(english_punctuations)

#english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',"''",'``',"'s","-","--",'–']
#stop_words=[]
#stop_words.extend(english_punctuations)
import random  
import numpy as np
# freq:1/100,  keep prob=0.1
# freq:1/500,  keep prob=sqrt(3)/2:  0.866
# freq:1/1000, keep prob=sqrt(2) 
def to_var(list_input,use_cuda,evaluate=False):
    # if evaluate, volatile=True, no grad be computed
    if use_cuda:
        output=Variable(torch.LongTensor(list_input),volatile=evaluate).cuda()
    else:
        output=Variable(torch.LongTensor(list_input),volatile=evaluate)
    return output

# 1  build w+phrase dict       unk 0   /     freq: word_id 2 freq      no unk
def build_dict(args,samples,test_samples,most_common=2e5,min_count=0):
    count=Counter()
    for ex_id,sample in enumerate(samples):
        # sample['all_Q_tokens']['phrase_tokens']
        #print(ex_id)
        phrase_tokens=[w.lower() for w in sample[args.Q_type]['phrase_tokens']]
        #phrase_tokens=[w.lower() for w in sample['all_Q_tokens']['phrase_tokens']]
        #phrase_tokens=[w.lower() if isinstance(w,str) else w[-1].lower() for w in sample['all_Q_tokens']['phrase_tokens'] ]
        count.update(phrase_tokens)   # word_dict :lower w
        count.update([sample['triple'][1][0].lower()])
        count.update([sample['triple'][1][2].lower()])
        
    for ex_id,sample in enumerate(test_samples):
        for doc_id,doc in enumerate(sample['document']):
            cans=[w.lower() for w in sample['raw_can'][doc_id]]
            count.update(cans)   # word_dict :lower w
            
    if most_common!=0:
        count=dict(count.most_common(most_common))
    
    word_dict = {}                              # w -->idx   word['w']=1    UNK-->0
    id2f = {}                                   # idx--> count 
    #  min_count = 1
    for word in count:
        #if count[word] >= min_count and word not in stop_words:    # drop rare words to unique
        if count[word] > min_count:  
       # if count[word] > min_count and word not in stop_words:       # drop rare words to unique
            id_ = len(word_dict) + 1
            word_dict[word] = id_       
            id2f[str(id_)] = count[word]
    word_dict['UNK'] = 0
    id2f[str(0)]=0
    id2f = OrderedDict(sorted(id2f.items()))    
    return word_dict,id2f
    
# 2 Q_dict, index after V
def build_Q_dict(args,samples,V):
    Q2label=dict()
    e_ids=set()
    r_ids=set()
    triple_set=set()
    for sample in samples:
        e1_id,p_id,e2_id=sample['triple'][0]
        e_ids.add(e1_id)
        e_ids.add(e2_id)
        r_ids.add(p_id)
        triple_set.add((e1_id,p_id,e2_id))
        Q2label[e1_id]=sample['E_info']['e1_mention'].lower()           # Q2label:lower label,use mention
        Q2label[e2_id]=sample['E_info']['ans_mention'].lower()
#        Q2label[e1_id]=sample['e1'].lower()           # Q2label:lower label
#        Q2label[e2_id]=sample['answear'].lower()
    Q_dict=dict(zip(list(e_ids),list(range(V,len(e_ids)+V))))
    r_dict=dict(zip(list(r_ids),list(range(len(r_ids)))))
    return Q_dict,r_dict,list(triple_set),Q2label

#  3 name_graph replace Q by name
def name_graph(triple_set,Q_dict,r_dict,Q2label,word_dict):
    name_KB=set()
    KB=set()
    for Q1,P,Q2 in triple_set:
        flag_1=False
        flag_2=False
        KB.add((Q_dict[Q1],r_dict[P],Q_dict[Q2]))
        if Q2label[Q1] in word_dict:
            w1_id=word_dict[Q2label[Q1]]                # Q's mention, correspond in word in text
            name_KB.add((w1_id,r_dict[P],Q_dict[Q2]))
            flag_1=True
        if Q2label[Q2] in word_dict:
            w2_id=word_dict[Q2label[Q2]]                # Q's name
            name_KB.add((Q_dict[Q2],r_dict[P],w2_id))
            flag_2=True
        if flag_1 and flag_2:
            name_KB.add((w1_id,r_dict[P],w2_id))
    return list(name_KB),list(KB)

def sub_sample(subsample, word, corpus_size, freq):
    if subsample > 0: # subsample=
        #print(word)
        word_freq = freq[str(word)]
        # prob of dropping context word c 
        if word_freq==0:
            print(word)
            raise ValueError
        keep_prob = (np.sqrt(word_freq / (subsample * corpus_size)) + 1) * subsample * corpus_size / word_freq
        if random.random() > keep_prob:
            return True  # drop
    return False
    
class StableBCELoss(nn.modules.Module):
       def __init__(self):
             super(StableBCELoss, self).__init__()
       def forward(self, input, target):
             neg_abs = - input.abs()
             loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
             return loss.mean()
         
def evaluate(samples,model,V,E,Q_dict,r_dict,args):
    entity_embedding=model.state_dict()['embedding.weight'][V:].cpu().numpy()
    assert len(entity_embedding)==E
    relation_embedding=model.state_dict()['r_embedding.weight'].cpu().numpy()
    exact_match=0
    exact_match3=0
    exact_match10=0
    exclude_self=0
    margin=1 if args.train_mode=='ptranse' else args.margin
    
    # Q_dict:  eid 2 index  (V+E)
    eids=list(Q_dict.keys())   # all candidate ex_ids
    for i,sample in enumerate(samples):
        e1_id,p_id,ans_id=sample['triple'][0]
        #e1_id=sample['e1_id']
        #p_id=sample['p_id']
        #ans_id=sample['ans_id']
        
        # and thier pos
        pre_indexs=np.array(list(Q_dict.values()))-V
        assert len(pre_indexs)==E
        
        if Q_dict.get(e1_id)!=None and r_dict.get(p_id)!=None:
            # pos in embeddings
            e1_idx=Q_dict[e1_id]-V
            p_idx=r_dict[p_id]
            final_scores=margin-0.5*np.sum((entity_embedding[e1_idx]+relation_embedding[p_idx]-entity_embedding[pre_indexs])**2,1)  # bigger, better
            new_index=np.argsort(-final_scores,)
            predictions=list(np.array(eids)[new_index])
            prediction=predictions[0]
            correct=prediction in ans_id
            exact_match+=correct
            
            prediction=predictions[1] if prediction[0]==e1_id and len(predictions)>1 else predictions[0]
            correct=prediction in ans_id
            exclude_self+=correct
            
            correct3=any([p for p in predictions[:3] if p in ans_id])
            exact_match3+=correct3
            
            correct10=any([p for p in predictions[:10] if p in ans_id])
            exact_match10+=correct10
            
            if len([e for e in ans_id if e in eids])!= len(ans_id):
                print('what!!!, all ans should in train e_ids')
                print(ans_id)
                raise ValueError
                break                    
    total=len(samples)
    #exact_match_exist_rate = 100.0 * exact_match_exist/ total_have
    exact_match_rate = 100.0 * exact_match / total
    exact_match_rate3 = 100.0 * exact_match3 / total
    exact_match_rate10 = 100.0 * exact_match10 / total
    exclude_self_rate=100.0 * exclude_self / total
    return exact_match,exclude_self_rate,total,exact_match_rate,exact_match_rate3,exact_match_rate10,total
import os    
def make_dir(my_dir):
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)
    else:
        pass

from collections import deque
# repeatly output batch in one one doc
def text_batch(args,samples,word_dict,id2f,batch_size,dp_stopwd=True,skip_win=3,sample_rate=1e-3):
    global pos_in_ex               # last added element's pos in this sample
    global ex_id
    new_sample=False
    newepoch=False                 # if another epoch
    span = 2 * skip_win + 1        # [skip_window target skip_window]
    buffer = deque(maxlen=span)
    L=len(samples)
    
    sample=samples[ex_id]
    doc=sample[args.Q_type]['phrase_tokens']
    #doc=sample['all_Q_tokens']['phrase_tokens']
    doc=list(map(lambda x:word_dict.get(x.lower(),0),doc))
    if dp_stopwd:
        doc=[w for w in doc if w!=0]                   # index
    # init smaple
    while (len(doc)==0):
        if (ex_id+1)>=L:
            newepoch=True
        ex_id=(ex_id+1)%L
        sample=samples[ex_id]
        doc=sample[args.Q_type]['phrase_tokens']
        #Sdoc=sample['all_Q_tokens']['phrase_tokens']
        doc=list(map(lambda x:word_dict.get(x.lower(),0),doc))
        if dp_stopwd:
            doc=[w for w in doc if w!=0]
    centers=[]
    contexts=[]
    
    # span 
    for i in range(span):
        buffer.append(doc[(pos_in_ex+i)%len(doc)])
    if pos_in_ex==len(doc)-1:                     # if need new sample in this buffer
        new_sample=True
        
    V=len(word_dict)
    while len(centers)<batch_size:
        # print(buffer)
        center=buffer[skip_win]
        context=list(range(span))
        context.remove(skip_win)
        for i in context:
            c=buffer[i]                              # word_index                              
            if sub_sample(sample_rate,c,V,id2f):
                continue
            centers.append(center)
            contexts.append(c)
       
        if new_sample==False:
            pos_in_ex=pos_in_ex+1
            buffer.append(doc[(pos_in_ex+span-1)%len(doc)]) #  if need new sample in next buffer
            if pos_in_ex==len(doc)-1:
                new_sample=True
        else:
            if (ex_id+1)>=L:
                newepoch=True
            ex_id=(ex_id+1)%L
            sample=samples[ex_id]
            doc=sample[args.Q_type]['phrase_tokens']
            #doc=sample['all_Q_tokens']['phrase_tokens']
            doc=list(map(lambda x:word_dict.get(x.lower(),0),doc))
            if dp_stopwd:
                doc=[w for w in doc if w!=0]  
            while (len(doc)==0 or len(doc==1)):
                if (ex_id+1)>=L:
                    newepoch=True
                ex_id=(ex_id+1)%L
                sample=samples[ex_id]
                doc=sample[args.Q_type]['phrase_tokens']
                #doc=sample['all_Q_tokens']['phrase_tokens']
                doc=list(map(lambda x:word_dict.get(x.lower(),0),doc))
                if dp_stopwd:
                    doc=[w for w in doc if w!=0]    
            buffer = deque(maxlen=span)
            # new sample, reset pos (but doc never change)
            pos_in_ex=0
            for i in range(span):
                buffer.append(doc[(pos_in_ex+i)%len(doc)])
            if pos_in_ex==len(doc)-1: # pos_in_ex+skip_win>=len(doc)-1
                new_sample=True
            else:
                new_sample=False
    return centers,contexts,newepoch
