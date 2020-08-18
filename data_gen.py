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

def generate_citizen_table(n=2000):
    citizen = pd.DataFrame()
    citizen['age'] = np.random.poisson( lam=30, size=n )
    citizen['gender'] = np.random.binomial( 1, 0.5, size=n )
    citizen['income'] = np.random.uniform(low=10,high=100, size=n)
    return citizen

def generate_lender_table(n=20):
    lender = pd.DataFrame()
    lender['asset'] = np.random.poisson( lam=1500, size=n )
    return lender

def generate_locality_table(n=15):
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

def generate_borrow_groups(citizens,residency_map,n=125):
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
                    net = net.append([[i,j],[j,i]],ignore_index=True)
    return net

def generate_loans(b_idx,lenders):
    lender_idx = lenders.index
    loan_map = np.random.choice(lender_idx,size=len(b_idx),replace=True)
    loans = pd.DataFrame(loan_map,index=b_idx)
    return loans, loan_map

def get_clique(array):
    clique = []
    for i in array:
        for j in array:
            if i!=j:
                clique.append([i,j])
                # clique.append([j,i])
    return pd.DataFrame(clique)

def add(edges,new_edges):
    edges_1 = edges.copy(deep=True)
    for i in new_edges.index:
        if not (edges_1 == new_edges.loc[i]).all(1).any():
            edges_1 = edges_1.append(new_edges.loc[i],ignore_index=True)
    return edges_1

def intervention(b_idx,borrowing_groups_rel,social_net):
    for i in b_idx:
        bg = borrowing_groups_rel.loc[borrowing_groups_rel[0]==i,1].to_numpy()
        clique = get_clique(bg)
        social_net = add(social_net[[0,1]],clique)
    return social_net

def generate_repayment(borrowing_groups,borrowing_groups_rel,loans,social_net):
    a0, a1 = (0.1,250)
    rr = []
    for i in borrowing_groups.index:
        bg = borrowing_groups_rel.loc[borrowing_groups_rel[0]==i,1]
        potential_edges = get_clique(bg.to_numpy())
        edges = sum([ ((social_net[[0,1]] == potential_edges.loc[i]).all(1).any()) for i in potential_edges.index ])
        rr.append(max(0,a0*100*(edges/len(potential_edges)) + a1/(loans.loc[i,'interest']+1) ))
    return rr
    
