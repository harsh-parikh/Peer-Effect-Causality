#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 01:37:25 2020

@author: harspari
"""

from data_gen import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#STRUCTURAL EQN
 
'''
social_net(C1,C2) <- age[C1], age[C2], income[C1], income[C2], residence[C1], residence[C2]
borrowing_group_rel(BG,C) <- residence[C]

capital[BG] <- income[C], age[C] where borrowing_group_rel(BG,C)
loan_amount[BG] <- size[BG], capital[BG]
loan_interest[BG] <- asset[L] where loan(BG,L)

repayment[BG] <- loan_interest[BG], social_net(C1,C2) where borrowing_group_rel(BG,C1) and borrowing_group_rel(BG,C2)
'''

def learn_capital(borrowing_groups, borrowing_groups_rel, citizens):
    b_idx = borrowing_groups.index
    Y = borrowing_groups['capital']
    X = pd.DataFrame()
    for i in b_idx:
        c_idx = borrowing_groups_rel.loc[borrowing_groups_rel['borrowing_group']==i,'citizen']
        df = citizens.loc[c_idx][['age','income']]
        X = X.append(df.mean(axis=0),ignore_index=True)
    rf = RandomForestRegressor().fit(X, Y)
    return lambda x: rf.predict(x)

def learn_amount(borrowing_groups, borrowing_groups_rel, loans):
    Y = loans['amount']
    X = pd.DataFrame()
    b_idx = borrowing_groups.index
    size = []
    for i in b_idx:
        size.append(len(borrowing_groups_rel.loc[borrowing_groups_rel['borrowing_group']==i,'citizen']))
    borrowing_groups['size'] = size
    X['size'] = size
    X['capital'] = borrowing_groups['capital']
    rf = RandomForestRegressor().fit(X, Y)
    return lambda x: rf.predict(x)

def learn_interest(loans, lenders):
    Y = loans['interest']
    X = lenders.loc[loans['lender']]
    rf = RandomForestRegressor().fit(X, Y)
    return lambda x: rf.predict(x)

def learn_repayment(borrowing_groups, loans, borrowing_groups_rel, social_net):
    Y = borrowing_groups['repayment']
    X = pd.DataFrame()
    X['interest'] = loans['interest']
    b_idx = borrowing_groups.index
    K = []
    for i in b_idx:
        c_idx = borrowing_groups_rel.loc[borrowing_groups_rel['borrowing_group']==i,'citizen']
        tuples = get_clique(c_idx.to_numpy())
        k = np.mean([ ((social_net.to_numpy() == tuples.loc[i].to_numpy()).all(1).any()) for i in tuples.index ])
        K.append(k)
    X['social_net'] = K
    rf = LinearRegression().fit(X, Y)
    return lambda x: rf.predict(x)

def estimate_repayment( f, borrowing_groups, loans, borrowing_groups_rel, social_net ):
    X = pd.DataFrame()
    X['interest'] = loans['interest']
    b_idx = borrowing_groups.index
    K = []
    for i in b_idx:
        c_idx = borrowing_groups_rel.loc[borrowing_groups_rel['borrowing_group']==i,'citizen']
        tuples = get_clique(c_idx.to_numpy())
        k = np.mean([ ((social_net.to_numpy() == tuples.loc[i].to_numpy()).all(1).any()) for i in tuples.index ])
        K.append(k)
    X['social_net'] = K
    Y = f(X)
    return X, Y


## DATA GENERATION
np.random.seed(0)
citizens = generate_citizen_table()
lenders = generate_lender_table()
localities = generate_locality_table()

residency, residency_map = generate_residency(citizens, localities)
borrower_idx, borrowing_groups_rel = generate_borrow_groups(citizens,residency_map)
social_net = generate_social_network(citizens,residency_map)
loans, loan_map = generate_loans(borrower_idx,lenders)
borrowing_groups = pd.DataFrame( index = borrower_idx )

#SETTING UP RELATIONSHIPS
borrowing_groups['capital'] = [np.mean([ citizens.loc[i,'income']*10/citizens.loc[i,'age'] 
                               for i in borrowing_groups_rel.loc[borrowing_groups_rel[0] == j,1].to_numpy() ]) 
                               for j in borrower_idx]
loans['amount'] = [len(borrowing_groups_rel.loc[borrowing_groups_rel[0] == j,1])*borrowing_groups['capital'].loc[j]/10 
                   for j in loans.index]
loans['interest'] = [ np.exp(-lenders.loc[loans[0].loc[j]]['asset']/500)*100 for j in loans.index ]


borrowing_groups['repayment'] = generate_repayment(borrowing_groups,borrowing_groups_rel,loans,social_net)

# INTERVENTION
social_net_1 = intervention( borrower_idx, borrowing_groups_rel, social_net)

#TRUE POTENTIAL OUTCOME (do not use it for anything except comparing the output)
borrowing_groups_1 = borrowing_groups.copy(deep=True)
borrowing_groups_1['repayment'] = generate_repayment(borrowing_groups,borrowing_groups_rel,loans,social_net_1)

#RENAME COLUMNS
loans.rename(columns={0:'lender'},inplace=True)
borrowing_groups_rel.rename(columns={0:'borrowing_group',1:'citizen'},inplace=True)
social_net.rename(columns={0:'citizen_0',1:'citizen_1'},inplace=True)
social_net_1.rename(columns={0:'citizen_0',1:'citizen_1'},inplace=True)


## LEARNING CONDITIONAL DISTRIBUTIONS (STRUCTURAL EQNS)
f_capital = learn_capital(borrowing_groups, borrowing_groups_rel, citizens) #f_capital: s(age x income) -> capital
f_amount = learn_amount(borrowing_groups, borrowing_groups_rel, loans) #f_amount: size x capital -> amount
f_interest = learn_interest(loans, lenders) #f_interest: asset -> interest

f_repayment = learn_repayment(borrowing_groups, loans, borrowing_groups_rel, social_net) #f_repayment: interest x s(social_net) -> repayment

#ESTIMATING POTENTIAL OUTCOMING
X0, repayment0 = estimate_repayment( f_repayment, borrowing_groups, loans, borrowing_groups_rel, social_net )
X1, repayment1 = estimate_repayment( f_repayment, borrowing_groups, loans, borrowing_groups_rel, social_net_1 )

sns.set()

fig = plt.figure()
plt.scatter(X0['social_net'],repayment0)
plt.scatter(X0['social_net'],borrowing_groups['repayment'])
plt.legend(['estimated','truth'])
plt.ylabel('Repayment Rate (%)')
plt.xlabel('Summarized(Social Network Attribute)')
fig.savefig('Figures/socialNet_vs_repayment0.png')

fig = plt.figure()
plt.plot([0,100],[0,100],c='black')
plt.scatter(borrowing_groups['repayment'],repayment0)
plt.scatter(borrowing_groups_1['repayment'],repayment1)
plt.legend(['x=y','control','treated'])
plt.ylabel('Est. Repayment Rate (%)')
plt.xlabel('True Repayment Rate (%)')
fig.savefig('Figures/estimated_vs_true_repayment.png')

fig = plt.figure()
plt.boxplot(repayment0,positions=[0])
plt.boxplot(repayment1,positions=[1])
plt.boxplot(borrowing_groups_1['repayment'],positions=[2])
plt.xticks(ticks=[0,1,2],
           labels=['Repayment Rate\n (before intervention)',
                   'Est. Repayment Rate\n (after intervention)',
                   'True Repayment Rate\n (after intervention)'], rotation=0)
plt.ylabel('Repayment Rate (%)')
plt.tight_layout()
fig.savefig('Figures/boxplot_estimation_repayment.png')
