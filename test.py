#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 20:31:21 2018

@author: xuweijia
"""

import torch
import json
import numpy as np
import argparse
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

parser = argparse.ArgumentParser(description='test ms embeddings')
parser.register('type', 'bool', str2bool)
parser.add_argument('--model_file',type=str, default='model_dir/',  help='model_file')
# parser.add_argument('--test_file', type=str, default='/data/disk1/private/xuweijia/new_NER_data/up_{}/test_contain_e.json'.format(ee),help='file to test')
# parser.add_argument('--test_file', type=str, default='data/final_test/dev_contain_e_valid_cands_tokenized_all.json',help='file to test')

args = parser.parse_args()
args.model_file=args.model_file+'model_ptranse_usembedFalse_embsize150_embnameglove_delayFalse_delaye0_delayb0_docB72_kbB64_docgap1_skip3_mostcom0_mincount0_nneg10.pkl'

pre='/media/xuweijia/00023F0D000406A9/my_drqa_up/'
mode='dev'
args.test_file=pre+'data/final_test/{}_contain_e_valid_cands_tokenized_all.json'.format(mode)



paras=torch.load(args.model_file)
embeddings=paras['state_dict']
V=len(paras['word_dict'])
Q_dict=paras['Q_dict']          # e_id 2 index
r_dict=paras['r_dict']          # e_id 2 index
args_old=paras['args']
entity_embedding=embeddings['embedding.weight'][V:].cpu().numpy()
relation_embedding=embeddings['r_embedding.weight'].cpu().numpy()
exact_match=0
exact_match3=0
exact_match10=0

exclude_self=0
margin=1 if args_old.train_mode=='ptranse' else args_old.doc_margin

with open(args.test_file,'r') as f:
    samples=json.load(f)
for i,sample in enumerate(samples):
    #e1_id=sample['e1_id']
    #p_id=sample['p_id']
    #ans_id=sample['ans_id']
    e1_id,p_id,ans_id=sample['triple'][0]
    # all candidate ex_ids
    eids=list(Q_dict.keys()) 
    # and thier pos
    pre_indexs=np.array(list(Q_dict.values()))-V
    
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
        
        correct3=len(([p for p in predictions[:3] if p in ans_id]))!=0
        #correct3=any([p for p in predictions[:3] if p in ans_id])
        exact_match3+=correct3
        
        correct10=len(([p for p in predictions[:10] if p in ans_id]))!=0
        # correct10=any([p for p in predictions[:10] if p in ans_id])
        exact_match10+=correct10
        
        if len([e for e in ans_id if e in eids])!= len(ans_id):
            print('what!!!, all ans should in train e_ids')
            print(i)
            raise ValueError
            break
total=i+1
#exact_match_exist_rate = 100.0 * exact_match_exist/ total_have
exact_match_rate = 100.0 * exact_match / total
exact_match_rate3 = 100.0 * exact_match3 / total
exact_match_rate10 = 100.0 * exact_match10 / total
exclude_self_rate=100.0 * exclude_self / total
print({'mode':mode},{'model_name':args.model_file})
print({'exact_match': exact_match},{'total:':total}) 
print({'exact_match_rate': exact_match_rate})  
print({'exclude_self_rate': exclude_self_rate})  
print({'exact_match_rate3': exact_match_rate3})  
print({'exact_match_rate10': exact_match_rate10})  
