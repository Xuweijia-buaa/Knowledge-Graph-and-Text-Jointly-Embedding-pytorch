#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 12:42:33 2018

@author: xuweijia
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import StableBCELoss,to_var


# not use self.cuda !!!!
# sampler (iterare class) no ()
# stored int key (json) --> str 
class MS_model(nn.Module):
    def __init__(self,args,word_dict,id2f,vocab_size,entity_size,r_size,n_neg):
        super(MS_model, self).__init__()
        self.args=args
        self.V=vocab_size
        self.E=entity_size
        self.R=r_size
        self.I=self.V+self.E
        self.freq=id2f
        self.h=args.embedding_size
        self.p=np.array(list(self.freq.values()))**(3/4)
        self.mysampler=torch.utils.data.sampler.WeightedRandomSampler(self.p,n_neg,replacement=False)    # sample from 0-len(weight)
        self.mode=args.train_mode
        
        self.embedding=nn.Embedding(self.I,self.h,padding_idx=0)
        self.r_embedding=nn.Embedding(self.R,self.h)
        self.context_embedding=nn.Embedding(self.I,self.h,padding_idx=0)                      #  context words embedding
        self.use_cuda=args.use_cuda
        self.bceloss=StableBCELoss()
        embedding = self.embedding.weight.data
        if args.using_embed:
            name=args.name
            n_in_embed=0
            if name=='word2vec_glove':
                skip_flag=True
            elif name=='glove':
                skip_flag=False
            print(50*'*'+'loading embedding'+50*'*')
            with open(args.embedding_file) as f:
                n=0
                for line in f:
                    parsed = line.rstrip().split(' ')
                    if n==0 and skip_flag:
                        n+=1
                        continue
                    assert(len(parsed) == self.h + 1)
                    # w in embed_file
                    w = parsed[0]
                    if w in word_dict:
                        n_in_embed+=1
                        vec=parsed[1:]
                        vec = torch.Tensor([float(i) for i in vec])
                        embedding[word_dict[w]].copy_(vec)
                    n+=1    
            print('words in embeddings / all words :{}/{}'.format(n_in_embed,self.V))
                        
    #def forward(self,centers,contexts,KB_batch,Q_centers,Q_contexts,name_KB_batch,n_neg=10,margin=7):
    def forward(self,inputs):
        n_neg=self.args.n_neg
        margin=self.args.margin if self.mode=='ptranse' else self.args.doc_margin
        loss=0
        # 1 KB loss
        if inputs.get('KB_batch')!=None:
            KB_batch=inputs.get('KB_batch')

            KB_batch=np.array(KB_batch)
            pos_h=KB_batch[:,0]     # (B,3)
            pos_r=KB_batch[:,1]
            pos_t=KB_batch[:,2]
            
            pos_h=to_var(pos_h,self.use_cuda)
            pos_r=to_var(pos_r,self.use_cuda)
            pos_t=to_var(pos_t,self.use_cuda)
            
            pos_h_embedding=self.embedding(pos_h)    # h: B,h
            pos_r_embedding=self.r_embedding(pos_r)  # r: B,h
            pos_t_embedding=self.embedding(pos_t)    # t: B,h
            
            neg_h=np.random.choice(list(range(self.V,self.V+self.E)),size=n_neg,replace=False)
            neg_h_embedding=self.embedding(to_var(neg_h,self.use_cuda))                               # neg_h: n,h                     
            neg_t=np.random.choice(list(range(self.V,self.V+self.E)),size=n_neg,replace=False)
            neg_t_embedding=self.embedding(to_var(neg_t,self.use_cuda))                               # neg_t: n,h                     
            neg_r=np.random.choice(self.R,size=n_neg,replace=True)
            neg_r_embedding=self.r_embedding(to_var(neg_r,self.use_cuda))                             # neg_r: n,h
     
            pos_score=margin-0.5*torch.sum((pos_h_embedding+pos_r_embedding-pos_t_embedding)**2,1)       # B,1    z=b-0.5|h+r-t|**2  P=σ(z(h,r,t))
            # h'+r-t    B,n_neg,h --> B,n_neg
            neg_h_score=margin-0.5*torch.sum((neg_h_embedding.unsqueeze(0).expand(len(pos_h),n_neg,self.h) + (pos_r_embedding-pos_t_embedding).unsqueeze(1).expand(len(pos_h),n_neg,self.h))**2,-1).squeeze(-1)
            neg_t_score=margin-0.5*torch.sum(((pos_h_embedding+pos_r_embedding).unsqueeze(1).expand(len(pos_h),n_neg,self.h)-neg_t_embedding.unsqueeze(0).expand(len(pos_h),n_neg,self.h))**2,-1).squeeze(-1)
            neg_r_score=margin-0.5*torch.sum(((pos_h_embedding-pos_t_embedding).unsqueeze(1).expand(len(pos_h),n_neg,self.h)+neg_r_embedding.unsqueeze(0).expand(len(pos_h),n_neg,self.h))**2,-1).squeeze(-1)
            # loss
            target=to_var(torch.ones(pos_score.size()).long(),self.use_cuda).float()
            target_neg=to_var(torch.zeros(neg_h_score.size()).long(),self.use_cuda).float()
            loss_KB=self.bceloss(pos_score,target)*3 +(self.bceloss(neg_h_score,target_neg)+self.bceloss(neg_t_score,target_neg)+self.bceloss(neg_r_score,target_neg)) * n_neg
            # loss_KB=self.bceloss(pos_score,target)*3 +(self.bceloss(neg_h_score,target_neg)+self.bceloss(neg_t_score,target_neg)+self.bceloss(neg_r_score,target_neg))
            loss+=loss_KB
        
        if self.mode=='ptranse':
            return loss_KB
        
        # 2 text loss
        if inputs.get('context_batch')!=None:
            centers,contexts=inputs.get('context_batch')        
            centers=to_var(centers,self.use_cuda)
            contexts=to_var(contexts,self.use_cuda)
            ceter_embedding=self.embedding(centers)              # B,h
            context_embedding=self.context_embedding(contexts)   # B,h  context embedding
            # negative index
            neg_index=[]
            for i in self.mysampler:
                neg_index.append(i)
            neg_index=to_var(neg_index,self.use_cuda)                # n neg_samples index
            neg_embedding=self.context_embedding(neg_index)      # n,h  context embedding
            # pos score    
            pos_score=margin-0.5*torch.sum((ceter_embedding-context_embedding)**2,1)     # B,1     z=b-0.5|v-w|**2  P=σ(z(w , v))
            # neg score
            center_expand=ceter_embedding.unsqueeze(1).expand(len(centers),n_neg,self.h) # B,n,h      each n,h, all line is this ex's vec 
            neg_expand=neg_embedding.expand_as(center_expand)                            # B,n,h      all batches are same, ex's n neg vecs
            neg_score=margin-0.5*torch.sum((center_expand-neg_expand)**2,-1).squeeze(-1)            # B,n        each line: ex to n_neg score : z_neg1 z_neg2 ...  z_neg_nneg
            
            target=to_var(torch.ones(pos_score.size()).long(),self.use_cuda).float()
            target_neg=to_var(torch.zeros(neg_score.size()).long(),self.use_cuda).float()
            # yi  * log sigmoid(z)  +  (1-yi) * log (1-sigmoid(z))       
            loss_text=self.bceloss(pos_score,target) + self.bceloss(neg_score,target_neg)*n_neg # torch.mean(torch.log(1/(torch.exp(neg_score)+1)))  p(0)=1/(1+ex) *n_neg?            
            # loss_text=self.bceloss(pos_score,target) + self.bceloss(neg_score,target_neg)
            loss+=loss_text
        
        # 3 Q_text_loss
        if inputs.get('Q_context_batch')!=None:
            Q_centers,Q_contexts=inputs.get('Q_context_batch')        
        #if self.mode=='joint' or self.mode=='just_anchor':
            centers=to_var(Q_centers,self.use_cuda)
            contexts=to_var(Q_contexts,self.use_cuda)
            ceter_embedding=self.embedding(centers)              # B,h
            context_embedding=self.context_embedding(contexts)   # B,h  context embedding
            # negative index (just from V)
            neg_index=[]
            for i in self.mysampler:
                neg_index.append(i)
            neg_index=to_var(neg_index,self.use_cuda)                # n neg_samples index
            neg_embedding=self.context_embedding(neg_index)      # n,h  context embedding
            # pos score    
            pos_score=margin-0.5*torch.sum((ceter_embedding-context_embedding)**2,1)    #  B,1     z=b-0.5|v-w|**2  P=σ(z(w , v))
            # neg score
            center_expand=ceter_embedding.unsqueeze(1).expand(len(centers),n_neg,self.h) # B,n,h      each n,h, all line is this ex's vec 
            neg_expand=neg_embedding.expand_as(center_expand)                            # B,n,h      all batches are same, ex's n neg vecs
            neg_score=margin-0.5*torch.sum((center_expand-neg_expand)**2,-1).squeeze(-1)            # B,n        each line: ex to n_neg score : z_neg1 z_neg2 ...  z_neg_nneg
            # loss         yi  * log sigmoid(z)  +  (1-yi) * log (1-sigmoid(z))  
            target=to_var(torch.ones(pos_score.size()).long(),self.use_cuda).float()
            target_neg=to_var(torch.zeros(neg_score.size()).long(),self.use_cuda).float()
            loss_Qtext=self.bceloss(pos_score,target) + self.bceloss(neg_score,target_neg) * n_neg
            # loss_Qtext=self.bceloss(pos_score,target) + self.bceloss(neg_score,target_neg)
            loss+=loss_Qtext
             
         # 4 name KB loss
         # if self.mode=='joint' or self.mode=='just_name_KB':
        if inputs.get('KB_name_batch')!=None:
            name_KB_batch=inputs.get('KB_name_batch')     
            name_KB_batch=np.array(name_KB_batch)
            pos_h=name_KB_batch[:,0]     # (B,3)
            pos_r=name_KB_batch[:,1]
            pos_t=name_KB_batch[:,2]
            
            pos_h=to_var(pos_h,self.use_cuda)
            pos_r=to_var(pos_r,self.use_cuda)
            pos_t=to_var(pos_t,self.use_cuda)
            
            pos_h_embedding=self.embedding(pos_h)    # h: B,h
            pos_r_embedding=self.r_embedding(pos_r)  # r: B,h
            pos_t_embedding=self.embedding(pos_t)    # t: B,h
            
            neg_h=np.random.choice(self.I,size=n_neg,replace=False)
            neg_h_embedding=self.embedding(to_var(neg_h,self.use_cuda))                               # neg_h: n,h                     
            neg_t=np.random.choice(self.I,size=n_neg,replace=False)
            neg_t_embedding=self.embedding(to_var(neg_t,self.use_cuda))                               # neg_t: n,h                     
            neg_r=np.random.choice(self.R,size=n_neg,replace=True)
            neg_r_embedding=self.r_embedding(to_var(neg_r,self.use_cuda))                             # neg_r: n,h
     
            pos_score=margin-0.5*torch.sum((pos_h_embedding+pos_r_embedding-pos_t_embedding)**2,1)       # B,1    z=b-0.5|h+r-t|**2  P=σ(z(h,r,t))
            # h'+r-t    B,n_neg,h --> B,n_neg
            neg_h_score=margin-0.5*torch.sum((neg_h_embedding.unsqueeze(0).expand(len(pos_h),n_neg,self.h) + (pos_r_embedding-pos_t_embedding).unsqueeze(1).expand(len(pos_h),n_neg,self.h))**2,-1).squeeze(-1)
            neg_t_score=margin-0.5*torch.sum(((pos_h_embedding+pos_r_embedding).unsqueeze(1).expand(len(pos_h),n_neg,self.h)-neg_t_embedding.unsqueeze(0).expand(len(pos_h),n_neg,self.h))**2,-1).squeeze(-1)
            neg_r_score=margin-0.5*torch.sum(((pos_h_embedding-pos_t_embedding).unsqueeze(1).expand(len(pos_h),n_neg,self.h)+neg_r_embedding.unsqueeze(0).expand(len(pos_h),n_neg,self.h))**2,-1).squeeze(-1)
            # loss
            target=to_var(torch.ones(pos_score.size()).long(),self.use_cuda).float()
            target_neg=to_var(torch.zeros(neg_h_score.size()).long(),self.use_cuda).float()
            loss_name_KB=self.bceloss(pos_score,target)*3 +(self.bceloss(neg_h_score,target_neg)+self.bceloss(neg_t_score,target_neg)+self.bceloss(neg_r_score,target_neg)) * n_neg       
            # loss_name_KB=self.bceloss(pos_score,target)*3 +(self.bceloss(neg_h_score,target_neg)+self.bceloss(neg_t_score,target_neg)+self.bceloss(neg_r_score,target_neg))
            loss+=loss_name_KB
#        if self.mode=='joint':
#            return  loss_text+loss_KB+loss_Qtext+ loss_name_KB
#        elif self.mode=='just_name_KB':
#            return  loss_text + loss_KB + loss_name_KB
#        elif self.mode=='just_anchor':
#            return  loss_text + loss_KB + loss_Qtext
        return loss
#        

        
