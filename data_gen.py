#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:40:21 2020

@author: harspari
"""

import numpy as np
import pandas as pd
from itertools import product 
import scipy.special

def generate_citizen_table(n=500):
    citizen = pd.DataFrame()
    citizen['age'] = np.random.poisson( lam=30, size=n )
    citizen['gender'] = np.random.binomial( 1, 0.5, size=n )
    citizen['income'] = np.random.uniform(low=10,high=100, size=n)
    return citizen

def generate_lender_table(n=5):
    lender = pd.DataFrame()
    lender['asset'] = np.random.poisson( lam=1500, size=n )
    return lender

def generate_locality_table(n=10):
    locality = pd.DataFrame()
    locality['lat'] = np.random.uniform(21,25,n)
    locality['long'] = np.random.uniform(75,80,n)
    return locality

def generate_residency(citizens,localities):
    #randomly assign localities
    local_idx = localities.index
    citizen_idx = citizens.index
    residency_map = np.random.choice(local_idx,size=len(citizen_idx),replace=True)
    residency = pd.DataFrame( np.zeros((citizens.shape[0],localities.shape[0])),index=citizen_idx, columns=local_idx )
    for i in range(len(citizen_idx)):
        residency.loc[citizen_idx[i],residency_map[i]] = 1.0
    return residency, residency_map

def generate_borrow_groups(citizens,residency_map,n=20):
    df = citizens
    df['residency'] = residency_map
    borrow = pd.DataFrame()
    b_idx = np.arange(0,n)
    for i in range(n):
        j = np.random.choice(pd.unique(residency_map))
        df_j = df.loc[ df['residency'] == j ].loc[ df['age'] > 18 ]
        k = np.random.randint(3,9)
        idxs = np.random.choice( df_j.index, size=k )
        b_i = pd.DataFrame(product([i],idxs))
        if i==0:
            borrow = b_i
        else:
            borrow = borrow.append(b_i,ignore_index=True)
    return b_idx, borrow
    
def generate_social_network(citizens,residency_map):
    a0 = 0.125
    a1 = 0.25
    a2 = 0.1
    net = pd.DataFrame()
    for i in citizens.index:
        for j in citizens.index:
            if i!=j:
                p_ij = scipy.special.expit( -a0*np.abs(citizens['age'].loc[i]-citizens['age'].loc[j]) 
                              + a1*(residency_map[i]==residency_map[j]) 
                              - a2*np.abs(citizens['income'].loc[i]-citizens['income'].loc[j]) )
                a_ij = np.random.binomial(1, p_ij)
                if a_ij>0:
                    net = net.append([[i,j,p_ij],[j,i,p_ij]])
    return net
    
np.random.seed(0)
citizens = generate_citizen_table()
lender = generate_lender_table()
localities = generate_locality_table()
residency, residency_map = generate_residency(citizens, localities)
borrower_idx, borrowing_groups = generate_borrow_groups(citizens,residency_map)
social_net = generate_social_network(citizens,residency_map)
