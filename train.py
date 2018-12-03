#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 12:42:33 2018

@author: xuweijia
"""
import json
import os
from collections import deque
from utils import sub_sample,evaluate,build_dict,build_Q_dict,name_graph,make_dir # stop_words
from joint_model import MS_model

import torch
import torch.utils.data
from torch.nn.utils import clip_grad_norm
import random
import argparse
import numpy as np
import time
# text word batch: each batch only one doc/ 
def text_batch(samples,word_dict,id2f,batch_size,dp_stopwd=True,skip_win=3,sample_rate=1e-3):
    global pos_in_ex               # last added element's pos in this sample
    global ex_id
    new_sample=False
    newepoch=False                 # if another epoch
    span = 2 * skip_win + 1        # [skip_window target skip_window]
    buffer = deque(maxlen=span)
    L=len(samples)
    
    sample=samples[ex_id]
    #doc=sample['phrase_tokens']
    doc=sample[args.Q_type]['phrase_tokens']
    #doc=sample['all_Q_tokens']['phrase_tokens']
    doc=list(map(lambda x:word_dict.get(x.lower(),0),doc))
    if dp_stopwd:
        doc=[w for w in doc if w!=0]                   # index

    while (len(doc)==0 or len(doc)==1):
        if (ex_id+1)>=L:
            newepoch=True
        ex_id=(ex_id+1)%L
        sample=samples[ex_id]
        doc=sample[args.Q_type]['phrase_tokens']
        #doc=sample['all_Q_tokens']['phrase_tokens']
        doc=list(map(lambda x:word_dict.get(x.lower(),0),doc))
        if dp_stopwd:
            doc=[w for w in doc if w!=0]
    centers=[]
    contexts=[]
    
    # fill span    pos_in_ex:start pos
    for i in range(span):
        buffer.append(doc[(pos_in_ex+i)%len(doc)])
    if pos_in_ex==len(doc)-1:
        new_sample=True
        
    V=len(word_dict)
    while len(centers)<batch_size:
        # print(buffer)
        center=buffer[skip_win]
        context=list(range(span))
        context.remove(skip_win)
        for i in context:
            c=buffer[i]                              # word_index                              
            #print({'ex_id',ex_id},{'text_batch doc':doc})
            #print(sample['all_Q_tokens']['phrase_tokens'])
            if sub_sample(sample_rate,c,V,id2f):
                continue
            centers.append(center)
            contexts.append(c)
        
        if new_sample==False:
            pos_in_ex=pos_in_ex+1   # pos_in_ex: next start pos, but only add one in buffer, until pos get to doc end/  (final buffer d[n-1],d[0],d[win-1]..,d[span-2] 
            buffer.append(doc[(pos_in_ex+span-1)%len(doc)])                                        #                    (start buffer d[0],d[1],d[win]..,d[span-1] 
            if pos_in_ex==len(doc)-1:
                new_sample=True
        else:          
            if (ex_id+1)>=L:
                newepoch=True
            ex_id=(ex_id+1)%L
            pos_in_ex=0
            sample=samples[ex_id]
            doc=sample[args.Q_type]['phrase_tokens']
            #doc=sample['all_Q_tokens']['phrase_tokens']
            doc=list(map(lambda x:word_dict.get(x.lower(),0),doc))
            if dp_stopwd:
                doc=[w for w in doc if w!=0]  
            while (len(doc)==0 or len(doc)==1):
                if (ex_id+1)>=L:
                    newepoch=True
                ex_id=(ex_id+1)%L
                pos_in_ex=0
                sample=samples[ex_id]
                doc=sample[args.Q_type]['phrase_tokens']
                #doc=sample['all_Q_tokens']['phrase_tokens']
                doc=list(map(lambda x:word_dict.get(x.lower(),0),doc))
                if dp_stopwd:
                    doc=[w for w in doc if w!=0]    
            buffer = deque(maxlen=span)
            # span 
            for i in range(span):
                buffer.append(doc[(pos_in_ex+i)%len(doc)])
            if pos_in_ex==len(doc)-1: # pos_in_ex+skip_win>=len(doc)-1
                new_sample=True
            else:
                new_sample=False
    return centers,contexts,newepoch

# Q_text batch
def text_batch_Q(samples,word_dict,Q_dict,id2f,batch_size,skip_win,sample_rate):
    global pos_in_ex_Q
    global ex_id_Q
    new_sample=False
    span = 2 * skip_win + 1        # [skip_window target skip_window]
    buffer = deque(maxlen=span)
    L=len(samples)
    
    sample=samples[ex_id_Q]
        
    doc=sample[args.Q_type]['Q_tokens']
    #doc=sample['all_Q_tokens']['Q_tokens']
    doc=[w for w in doc if w.lower() in word_dict or w in Q_dict]  # words
    while (len(doc)==0 or len(doc)==1):
        pos_in_ex_Q=0
        ex_id_Q=(ex_id_Q+1)%L
        sample=samples[ex_id_Q]
        doc=sample[args.Q_type]['Q_tokens']
        #doc=sample['all_Q_tokens']['Q_tokens']
        doc=[w for w in doc if w.lower() in word_dict or w in Q_dict]  # words       
    centers=[]
    contexts=[] 
    # span 
    for i in range(span):
        buffer.append(doc[(pos_in_ex_Q+i)%len(doc)])
    if pos_in_ex_Q==len(doc)-1:
        new_sample=True
        
    while len(centers)<batch_size:
        # print(buffer)
        center=buffer[skip_win]
        center_id=word_dict[center.lower()] if word_dict.get(center.lower()) else Q_dict[center]

        context=list(range(span))
        context.remove(skip_win)
        for i in context:
            c=buffer[i]
            c_id=word_dict[c.lower()] if word_dict.get(c.lower()) else Q_dict[c]
            if center not in Q_dict: 
                continue
            if c.lower() in word_dict:
                #print({'ex_id',ex_id_Q},{'Q text_batch doc':doc})
                #print(sample['all_Q_tokens']['phrase_tokens'])
                if sub_sample(sample_rate,c_id,V,id2f): # word_index  
                    continue
            centers.append(center_id)
            contexts.append(c_id)
        
        if new_sample==False:
            pos_in_ex_Q=pos_in_ex_Q+1
            buffer.append(doc[(pos_in_ex_Q+span-1)%len(doc)])
            if pos_in_ex_Q==len(doc)-1:
                new_sample=True
        else:    
            pos_in_ex_Q=0
            ex_id_Q=(ex_id_Q+1)%L
            sample=samples[ex_id_Q]
            doc=sample[args.Q_type]['Q_tokens']
            #doc=sample['all_Q_tokens']['Q_tokens']
            #doc=sample['all_Q_tokens']['Q_tokens']
            doc=[w for w in doc if w.lower() in word_dict or w in Q_dict]
            while (len(doc)==0 or len(doc)==1):
                pos_in_ex_Q=0
                ex_id_Q=(ex_id_Q+1)%L
                sample=samples[ex_id_Q]
                doc=sample[args.Q_type]['Q_tokens']
                #doc=sample['all_Q_tokens']['Q_tokens']
                doc=[w for w in doc if w.lower() in word_dict or w in Q_dict]  # words 
            buffer = deque(maxlen=span)
            # span 
            for i in range(span):
                buffer.append(doc[(pos_in_ex_Q+i)%len(doc)])
            if pos_in_ex_Q==len(doc)-1: # pos_in_ex+skip_win>=len(doc)-1
                new_sample=True
            else:
                new_sample=False
    return centers,contexts
    
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

#  KB_delay True  KB_delay_epoch 3    KB_delay_batch 0
#  KB_delay True  KB_delay_epoch 3    KB_delay_batch 10
#  KB_delay False KB_delay_epoch 0    KB_delay_batch 0
#  using_embed True embedding_file /data/...txt  KB_delay False KB_delay_epoch 0    KB_delay_batch 0
#  --batch_size 128  --KB_batch_size 64 KB_delay False KB_delay_epoch 0    KB_delay_batch 0
def add_train_args(parser):
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    # global settinging
    # ptranse each epoch: 1054 *64 batch    KB_name 2049*64  batch
    # ptranse ( epoch 100  100000)  embed_size: 150       epoch 79  batch_size 64  80000   batch        49.061/48.877    never better
    # epoch 1  100000 batch         embed_size: 150       epoch 0   batch_size 72  107000/126000/ batch   44.26/44.40/44.58
    settings = parser.add_argument_group('Setting')
    settings.add_argument('--train_mode', type=str, default='ptranse' ,help="joint' 'just_anchor' 'just_name_KB' 'ptranse' ")    # 3 mode
    settings.add_argument('--ptranse_epochs', type=int, default=200,help='4000/10000/100000') 
    settings.add_argument('--epochs', type=int, default=50,help='40/80/800')
    settings.add_argument('--batch_size', type=int, default=72,help='36/72/144 batch size m*(2*skip_win)')    
    settings.add_argument('--KB_batch_size', type=int, default=64,help='32/64/128')      
    settings.add_argument('--doc_margin', type=int, default=1,help='1/3/5/7, compute z score')    # 1 ~ 
    settings.add_argument('--skip_win', type=int, default=3,help='3/4/5/10, half span size')      # 3 ~                           # win:   2,3,5,10                       
    settings.add_argument('--n_neg', type=int, default=10,help='5/10 each pos how many neg')      # 2 ~
    settings.add_argument('--Q_type', type=str, default='all_Q_tokens',help='all_Q_tokens/only_Q_tokens')
    
    settings.add_argument('--using_embed', type='bool', default=False,help='if use pretain word vec to warm-start with')
    settings.add_argument('--embedding_file', type=str,default='glove.840B.300d.txt',help=('embedding file'))
    settings.add_argument('--KB_delay', type='bool', default=False,help='delay KB train')    
    settings.add_argument('--KB_delay_epoch', type=int, default=0,help='delay KB train 0/3')  # KB delay, just train text.  then same time
    settings.add_argument('--KB_delay_batch', type=int, default=0,help='delay KB train 0/10') # text train 10 batch, KB train 10 batch

    settings.add_argument('--margin', type=int, default=1,help='1, compute z score') 
    settings.add_argument('--most_common', type=int, default=0,help='0/2e5')
    settings.add_argument('--min_count', type=int, default=0,help='0/5 drop rare word')                                      # n_neg: 5/10
    settings.add_argument('--dp_stopwd', type='bool', default=False,help='drop stop words')  
    settings.add_argument('--sample_rate', type=float, default=1e-3,help='1e-3/1e-5 drop sample rate during text batchify')
    runtime = parser.add_argument_group('Environment')       
    runtime.add_argument('--train_file', type=str,default='',help=('train file'))
    runtime.add_argument('--dev_file', type=str, default='',help='dev')
    runtime.add_argument('--test_file', type=str, default='',help='test')
    runtime.add_argument('--model_dir', type=str,default='model_dir',help=('store dict,model'))              # change only data/min_count change
    runtime.add_argument('--worddict_file', type=str,default='dict.json',help=('word dict and others file'))
    runtime.add_argument('--tripleset', type=str,default='tripleset.json',help=('tripleset'))                                  # 32/64
    runtime.add_argument('--use_cuda', type='bool', default=True,help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--random_seed', type=int, default=1314,help=('Random seed for all numpy/torch/cuda operations (for reproducibility)'))
    runtime.add_argument('--gpu', type=int, default=0,help='Run on a specific GPU')
    # model paras
    model_paras= parser.add_argument_group('Model_para')
    model_paras.add_argument('--embedding_size', type=int, default=300,help='embedding size')                                   # 100/150/300
    model_paras.add_argument('--step_size', type=float, default=0.025,help='0.025/0.01 sgd step size')
    model_paras.add_argument('--GRAD_CLIP', type=int, default=10,help='GRAD_CLIP')
    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--print_every', type=int, default=100,help='print every')
    save_load.add_argument('--check_file', type=str, default='temp',help='temp state file')
    save_load.add_argument('--eval_every', type=int, default=1000,help='eval and save')
    
# main 
if __name__ == "__main__":
    parser = argparse.ArgumentParser('MS_model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_train_args(parser)
    args = parser.parse_args()
    make_dir(args.model_dir)

    #pre='/home/xuweijia/my_drqa_up/'
    pre='/media/xuweijia/00023F0D000406A9/my_drqa_up/'
    #pre='/data/disk3/private/xuweijia/my_drqa_up/'    
    args.train_file=pre+'data/final_train/train_tokenized.json'
    args.dev_file=pre+'data/final_test/dev_contain_e_valid_cands_tokenized_all.json'
    args.test_file=pre+'data/final_test/test_contain_e_valid_cands_tokenized_all.json'
    
    #pre='/data/disk1/private/xuweijia/DrQA/data/embeddings/'
    #args.embedding_file=pre+args.embedding_file
    
    assert (args.batch_size)%(2*args.skip_win)==0
    #args.dev_file=pre+'data/final_test/dev_contain_e.json'
    #args.test_file=pre+'data/final_test/test_contain_e.json'
    if args.KB_delay and args.using_embed:
        args.KB_delay= not args.using_embed
    
    if args.train_mode!='ptranse' and args.KB_delay and args.KB_delay_epoch:
        args.epochs+=args.KB_delay_epoch
        
    args.worddict_file='dict_mostcom{}_mincount{}_{}.json'.format(args.most_common,args.min_count,args.Q_type)
    # seed
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    args.use_cuda = (args.use_cuda)  and  (torch.cuda.is_available())
    if args.use_cuda:
        torch.cuda.set_device(args.gpu)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.random_seed)
        
    with open(args.train_file,'r') as f:
         samples=json.load(f)
         #samples=samples[:300]
    with open(args.dev_file,'r') as f:
         dev_samples=json.load(f)   
         #dev_samples=dev_samples[:300]
    with open(args.test_file,'r') as f:
         test_samples=json.load(f) 
         #test_samples=test_samples[:300]
         
    word_dict_path=os.path.join(args.model_dir,args.worddict_file)  
    if os.path.exists(word_dict_path):
        with open(word_dict_path,'r') as f: 
            dicts=json.load(f) 
            word_dict,id2f,Q_dict,r_dict,triple_set,Q2label= dicts['word_dict'],dicts['id2fre'],dicts['Q_dict'],dicts['r_dict'],dicts['triple_set'],dicts['Q2label'] 
    else:
        word_dict,id2f=build_dict(args,samples,test_samples+dev_samples,args.most_common,args.min_count)
        Q_dict,r_dict,triple_set,Q2label=build_Q_dict(args,samples,len(word_dict))       # only need to change folder when change data/min_count
        dicts={'word_dict':word_dict,'id2fre':id2f,'Q_dict':Q_dict,'r_dict':r_dict,'triple_set':triple_set,'Q2label':Q2label}    
        with open(word_dict_path,'w') as f: 
            json.dump(dicts,f)   
            
    V=len(word_dict)             
    E=len(Q_dict)
    R=len(r_dict)
    print('V:{},E:{},R:{}'.format(V,E,R))
    # indexed KB,name_KB set  
    name_KB,KB=name_graph(triple_set,Q_dict,r_dict,Q2label,word_dict)
    n_t=len(KB)
    n_name_t=len(name_KB)
    KB_batches=list(zip(list(range(0,n_t,args.KB_batch_size)),list(range(args.KB_batch_size,n_t+args.KB_batch_size,args.KB_batch_size))))  # each batch start,end index
    KB_name_batches=list(zip(list(range(0,n_name_t,args.KB_batch_size)),list(range(args.KB_batch_size,n_name_t+args.KB_batch_size,args.KB_batch_size))))
    
    KB_bidx=0
    KB_name_bidx=0    # which batch
    
    name=args.embedding_file.split('/')[-1].split('.')[:-1][0]
    args.name=name
    if args.using_embed:
        if name=='word2vec_glove':
            args.embedding_size=100
        elif name=='glove':
            args.embedding_size=300
            
    args.model_file='model_{}_usembed{}_embsize{}_embname{}_delay{}_delaye{}_delayb{}_docB{}_kbB{}_docgap{}_skip{}_mostcom{}_mincount{}_nneg{}_clip{}_Qtype{}.pkl'\
       .format(args.train_mode,args.using_embed,args.embedding_size,name,args.KB_delay,args.KB_delay_epoch,args.KB_delay_batch,args.batch_size,args.KB_batch_size,\
               args.doc_margin,args.skip_win,args.most_common,args.min_count,args.n_neg,args.GRAD_CLIP,args.Q_type)
                              

    model=MS_model(args,word_dict,id2f,V,E,R,args.n_neg)

    if args.use_cuda:
        model.cuda()
    # opt = torch.optim.SGD(params=model.parameters(),lr=args.step_size,momentum=0.9)
    opt = torch.optim.Adam(params=model.parameters(), lr = 0.001)
    # opt = torch.optim.Adamax((params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8))
    # start training
    start = time.time()
    best_valid_acc= best_test_acc = 0
    n_KB=0                                  #  all batches in al epoches
    n_KB_name=0
    n_text=0
    n_Q_text=0
    # ptranse
    if args.train_mode=='ptranse':
        n_batches=len(KB_batches) 
        args.eval_every=n_batches>>1
        for epoch in range(args.ptranse_epochs): # each epoch            
            new_max=False
#            for param_group in opt.param_groups:
#                if param_group['lr']> 0.01:
#                   param_group['lr'] = param_group['lr']-0.001
            model.train()
            loss = n_examples = it = 0
            random.shuffle(KB)
            for KB_bidx in range(n_batches): # each batch
                start_p,end_p=KB_batches[KB_bidx]                       
                KB_batch=KB[start_p:end_p]              #  (B,3)
                inputs={}
                inputs['KB_batch']=KB_batch
                loss_=model(inputs)
                loss += loss_.cpu().data.numpy()[0] # numpy [1]
                n_KB+=1                                  #  all batches in al epoches
                it += 1
                opt.zero_grad()
                loss_.backward()
                clip_grad_norm(parameters= model.parameters(),max_norm=args.GRAD_CLIP)
                opt.step()    
                if it % args.print_every == 0:
                    spend = (time.time() - start) / 60
                    statement = "Epoch: {}, it: {} ".format(epoch, it)
                    statement += "loss: {:.3f},time: {:.1f}(m)".format(loss / args.print_every, spend)
                    print(statement)
        		         # save every print
                    params = {'state_dict': model.state_dict(),'word_dict': word_dict,'freq_dict':id2f,'Q_dict':Q_dict,'r_dict':r_dict,'Q2label':Q2label, 'args': args}
                    # torch.save(params, os.path.join(args.store_folder,args.check_file+'_mode{}_epoch{}_it{}_loss{}.pkl'\
                    #                                .format(args.train_mode,epoch,it,loss)))
                    del loss
                    loss = 0
                if it % args.eval_every == 0:
                    model.eval()
                    spend = (time.time() - start) / 60
                    exact_match,exclude_self_rate,total,exact_match_rate,exact_match_rate3,exact_match_rate10,total=evaluate(dev_samples,model,V,E,Q_dict,r_dict,args)
                    statement = "Epoch: {}, it: {} (max: {}), mode:dev".format(epoch, it, n_batches)
                    print(statement)
                    print('{}__embsize{}_nneg{}_KBbatch{}'.format(args.train_mode,args.embedding_size,args.n_neg,args.KB_batch_size))
                    print({'exact_match': exact_match},{'total:':total}) 
                    print({'exact_match_rate (%)': exact_match_rate}) 
                    print({'exclude_self_rate': exclude_self_rate})
                    print({'exact_match_rate3(%)': exact_match_rate3})  
                    print({'exact_match_rate10(%)': exact_match_rate10})    
        		         # save every print                 
                    if best_valid_acc < exact_match_rate and exact_match_rate>30:
                        best_valid_acc = exact_match_rate
                        new_max=True
                        # store best valid model
                        params = {'state_dict': model.state_dict(),'word_dict': word_dict,'freq_dict':id2f,'Q_dict':Q_dict,'r_dict':r_dict,'Q2label':Q2label, 'args': args}
                        torch.save(params, os.path.join(args.model_dir,args.model_file))
                    print("Best valid acc: {:.3f}, mode:{}".format(best_valid_acc,args.train_mode))
            # after epoch, test       
            model.eval()       
            exact_match,exclude_self_rate,total,exact_match_rate,exact_match_rate3,exact_match_rate10,total=evaluate(test_samples,model,V,E,Q_dict,r_dict,args)
            spend = (time.time() - start) / 3600
            statement = "Epoch: {}, mode:test, time: {:.1f}(m)".format(epoch,spend)
            print(statement)
            print('{}_embsize{}_nneg{}_KBbatch{}'.format(args.train_mode,args.embedding_size,args.n_neg,args.KB_batch_size))
            print({'exact_match': exact_match},{'total:':total}) 
            print({'exact_match_rate': exact_match_rate})  
            print({'exclude_self_rate': exclude_self_rate})
            print({'exact_match_rate3': exact_match_rate3})  
            print({'exact_match_rate10': exact_match_rate10})  
            if best_test_acc < exact_match_rate:
                best_test_acc = exact_match_rate
            print("Best test acc: {:.3f},mode:{}".format(best_test_acc,args.train_mode))
            print( 'batches: n_KB:{}'.format(n_KB))  
            #if not new_max:
            #    break
    # joint    
    else:
        epoch=0
        pos_in_ex=0
        ex_id=0
        pos_in_ex_Q=0
        ex_id_Q=0
        new_epoch=False  # new epoch
        flag_test=False  # when to test
        flag_text=True   # when delay batch, train text/KB
        count_batch=0    # count dealy batch
        # 100 doc, (24: 300 text/KB batch;   72:120 text/KB batch;      144: 61 text/KB batch
        new_max=False
        loss = n_examples = it = 0                           
        model.train()
        while (epoch <args.epochs): # 0,1,...n_epoch-1
            if new_epoch:
                epoch+=1
                new_max=False
                random.shuffle(KB)
                random.shuffle(name_KB)
                loss = n_examples = it = 0
                flag_test=True                            
                model.train()
    #            for param_group in opt.param_groups:
    #                if param_group['lr']> 0.01:
    #                   param_group['lr'] = param_group['lr']-0.001    
            if args.KB_delay and args.KB_delay_batch and epoch<args.KB_delay_epoch:
                 # train text/KB independently first
                 if flag_text:
                     inputs={}                             
                     centers,contexts,new_epoch=text_batch(samples,word_dict,id2f,args.batch_size,args.dp_stopwd,args.skip_win,args.sample_rate)  # control epoch number
                     inputs['context_batch']=centers,contexts 
                     n_text+=1
                     count_batch+=1
                     if count_batch==args.KB_delay_batch:
                         flag_text=False
                         count_batch=0
                 else:
                     inputs={} 
                     start_p,end_p=KB_batches[KB_bidx]                     
                     KB_batch=KB[start_p:end_p]              #  (B,3)  
                     inputs['KB_batch']=KB_batch
                     count_batch+=1
                     n_KB+=1                                  #  all batches in al epoches
                     KB_bidx=(KB_bidx+1)%len(KB_batches)  
                     if count_batch==args.KB_delay_batch:
                         flag_text=True 
                         count_batch=0
                         
            elif args.KB_delay and args.KB_delay_epoch and epoch<args.KB_delay_epoch: #epoch<args.KB_delay_epoch:
                    # just train text word
#==============================================================================
#                     inputs={}
#                     centers,contexts,new_epoch=text_batch(samples,word_dict,id2f,args.batch_size,args.dp_stopwd,args.skip_win,args.sample_rate)  # control epoch number
#                     inputs['context_batch']=centers,contexts 
#                     n_text+=1
#==============================================================================
                    
                    inputs={} 
                    start_p,end_p=KB_batches[KB_bidx]                     
                    KB_batch=KB[start_p:end_p]              #  (B,3)  
                    inputs['KB_batch']=KB_batch
                    n_KB+=1                                  #  all batches in all epoches
                    if KB_bidx+1==len(KB_batches):
                        new_epoch=True
                    KB_bidx=(KB_bidx+1)%len(KB_batches) 
                    
                
            else:
                inputs={}
                
                start_p,end_p=KB_batches[KB_bidx]                       
                KB_batch=KB[start_p:end_p]              #  (B,3)  
                inputs['KB_batch']=KB_batch
                #print(KB_batch[0])
                n_KB+=1                                  #  all batches in all epoches
                KB_bidx=(KB_bidx+1)%len(KB_batches) 
                                
                centers,contexts,new_epoch=text_batch(samples,word_dict,id2f,args.batch_size,args.dp_stopwd,args.skip_win,args.sample_rate)  # control epoch number
                inputs['context_batch']=centers,contexts
                n_text+=1

                if args.train_mode=='just_anchor' or args.train_mode=='joint':                                         
                    Q_centers,Q_contexts=text_batch_Q(samples,word_dict,Q_dict,id2f,args.batch_size,args.skip_win,args.sample_rate)  # indexed Q_text_batch                  (B,),(B,) 
                    inputs['Q_context_batch']=Q_centers,Q_contexts
                    n_Q_text+=1
                       
                if args.train_mode=='just_name_KB' or args.train_mode=='joint':
                    start_p,end_p=KB_name_batches[KB_name_bidx]                       
                    name_KB_batch=name_KB[start_p:end_p]    #  (B,3)
                    inputs['KB_name_batch']=name_KB_batch
                    n_KB_name+=1
                    KB_name_bidx=(KB_name_bidx+1)%len(KB_name_batches) 

            loss_=model(inputs)
            
            embed_before=model.state_dict()['embedding.weight'][V:].cpu().numpy()
            
            loss += loss_.cpu().data.numpy()[0] # numpy [1]
            it += 1
            opt.zero_grad()
            loss_.backward()
            clip_grad_norm(parameters= model.parameters(),max_norm=args.GRAD_CLIP)
            opt.step()
            
            embed_after=model.state_dict()['embedding.weight'][V:].cpu().numpy()
            
            embed_change=embed_after-embed_before
            
            if it % args.print_every == 0:
                spend = (time.time() - start) / 60
                statement = "Epoch: {}, it: {} ".format(epoch, it)
                statement += "loss: {:.3f},time: {:.1f}(m)".format(loss / args.print_every, spend)
                print(statement)
    		         # save every print
                params = {'state_dict': model.state_dict(),'word_dict': word_dict,'freq_dict':id2f,'Q_dict':Q_dict,'r_dict':r_dict,'Q2label':Q2label, 'args': args}
                del loss
                loss = 0
                           
            if it % args.eval_every == 0:
                model.eval()
                spend = (time.time() - start) / 60
                exact_match,exclude_self_rate,total,exact_match_rate,exact_match_rate3,exact_match_rate10,total=evaluate(dev_samples,model,V,E,Q_dict,r_dict,args)
                statement = "Epoch: {}, it: {} , mode:dev".format(epoch, it)
                print(statement)
                print('{}_usembed{}_embsize{}_embname{}_delay{}_delaye{}_delayb{}_docgap{}_skip{}_nneg{}_docB{}_kbB{}_mostw{}_drop{}_clip{}_Qtype{}'\
                     .format(args.train_mode,args.using_embed,args.embedding_size,name,args.KB_delay,args.KB_delay_epoch,args.KB_delay_batch,\
                             args.doc_margin,args.skip_win,args.n_neg,args.batch_size,args.KB_batch_size,args.most_common,args.min_count,args.GRAD_CLIP,args.Q_type))
                print({'exact_match': exact_match},{'total:':total}) 
                print({'exact_match_rate (%)': exact_match_rate})  
                print({'exclude_self_rate': exclude_self_rate})
                print({'exact_match_rate3(%)': exact_match_rate3})  
                print({'exact_match_rate10(%)': exact_match_rate10})    
    		         # save every print                 
                if best_valid_acc < exact_match_rate and exact_match_rate>20:
                    best_valid_acc = exact_match_rate
                    new_max=True
                    # store best valid model
                    params = {'state_dict': model.state_dict(),'word_dict': word_dict,'freq_dict':id2f,'Q_dict':Q_dict,'r_dict':r_dict,'Q2label':Q2label, 'args': args}
                    torch.save(params, os.path.join(args.model_dir,args.model_file))
                print("Best valid acc: {:.3f}, mode:{}".format(best_valid_acc,args.train_mode))
                model.train()
            if flag_test:  
                # after epoch, test       
                model.eval()       
                exact_match,exclude_self_rate,total,exact_match_rate,exact_match_rate3,exact_match_rate10,total=evaluate(test_samples,model,V,E,Q_dict,r_dict,args)
                spend = (time.time() - start) / 3600
                statement = "Epoch: {}, mode:test, time: {:.1f}(m)".format(epoch,spend)
                print(statement)
                print('{}_usembed{}_embsize{}_embname{}_delay{}_delaye{}_delayb{}_docgap{}_skip{}_nneg{}_docB{}_kbB{}_mostw{}_drop{}_clip{}_Qtype{}'\
                     .format(args.train_mode,args.using_embed,args.embedding_size,name,args.KB_delay,args.KB_delay_epoch,args.KB_delay_batch,\
                             args.doc_margin,args.skip_win,args.n_neg,args.batch_size,args.KB_batch_size,args.most_common,args.min_count,args.GRAD_CLIP,args.Q_type))
                print({'exact_match': exact_match},{'total:':total}) 
                print({'exact_match_rate': exact_match_rate})  
                print({'exact_match_rate3': exact_match_rate3})  
                print({'exclude_self_rate': exclude_self_rate})
                print({'exact_match_rate10': exact_match_rate10})  
                if best_test_acc < exact_match_rate:
                    best_test_acc = exact_match_rate
                print("Best test acc: {:.3f},mode:{}".format(best_test_acc,args.train_mode))
                print( 'batches:  n_KB:{}, n_KB_name:{},n_text:{},n_Q_text:{}'.format(n_KB, n_KB_name,n_text,n_Q_text))                             
                model.train() 
                flag_test=False
                '''
                if not new_max:
                    break
                '''

