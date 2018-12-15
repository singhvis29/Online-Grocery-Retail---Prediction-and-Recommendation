# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 13:00:34 2018

@author: singh
"""
#Loading libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
import os
import pandas as pd
#import seaborn as sns
import itertools as iter
from collections import Counter
import gc
from sklearn.decomposition import PCA
#import statistics

##Loading Datasets
products = pd.read_csv("products.csv",dtype={
        'product_id': np.uint16,
        'product_name': object,
        'aisle_id': np.uint8,
        'department_id': np.uint8})

departments = pd.read_csv("departments.csv",dtype={
        'department_id': np.uint16,
        'department_name': object})
aisles = pd.read_csv("aisles.csv",dtype={
        'aisle_id': np.uint16,
        'aisle_name': object})
orders = pd.read_csv("orders.csv", dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': object,
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})
order_products_train = pd.read_csv("order_products__train.csv", dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
order_products_prior = pd.read_csv("order_products__prior.csv", dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

#Preparing Datasets
print('preparing datasets..')
items  = pd.merge(left =pd.merge(left=products, right=departments, how='left', on = 'department_id'), right=aisles, how='left', on = 'aisle_id')
prior_data = orders.merge(order_products_prior,how='inner', left_on='order_id', right_on='order_id')
prior_data = prior_data.merge(items,how='inner', left_on='product_id', right_on='product_id')
train_data = orders.merge(order_products_train,how='inner', left_on='order_id', right_on='order_id')
train_data = train_data.merge(items,how='inner', left_on='product_id', right_on='product_id')

print('list of 2500 users')
ordersTrain=orders[orders['eval_set']=='train']
train_usr = pd.DataFrame(ordersTrain['user_id'].unique(), columns =['user_id'])
train_usr_sample =  train_usr.sample(2500, random_state=42).sort_values('user_id').reset_index(drop = True)
orders=orders[orders['user_id'].isin(train_usr_sample['user_id'])]

#Creating Training Dataset
print('creating training datasets')
train_data = prior_data.merge(train_usr_sample,how='inner', left_on='user_id', right_on='user_id')
train_data['user_id'].nunique()

order_product = train_data[['order_id','product_name']]

order_product['product_number'] = order_product.groupby(['order_id']).cumcount()+1
order_product = order_product.sort_values(by=['order_id','product_number'])
print('order-product')
print(order_product.head())

order_product = order_product.pivot(index='order_id', columns='product_number', values='product_name')

op_index = order_product.index.values
product_list =  order_product.apply(lambda x :'*_*'.join(x.fillna("xxx").astype(str)),1)
product_list.index = op_index
order_product['product'] = product_list

order_product['order_id'] = order_product.index
order_product = order_product[['order_id','product']]

op = order_product.values
for i in range(len(op)):
    l = list(set(op[i][1].split('*_*')))
    if ('xxx' in l):
        l.remove('xxx')
    l = list(iter.combinations(l,2))
    op[i][1] = l
    
op_df = pd.DataFrame(op, columns=['order_id','product_pairs'])
print('order-product pairs')
print(op_df.head())

# del datasets
del departments
del aisles
del orders
del order_products_train
del order_products_prior
del items
del order_product
del prior_data
gc.collect()

res = op_df.apply(lambda x: pd.Series(x['product_pairs']),axis=1).stack().reset_index(level=1, drop=True)
res.name = 'product_pair'

op_df = op_df.drop('product_pairs', axis=1).join(res)
op_df.head()

op_df['order_id']=op_df['order_id'].astype('int64')

del res
gc.collect()

train_data = train_data[['user_id','order_id']]

up_df = train_data.merge(op_df, how='inner', on='order_id')

up_df = up_df.dropna()

del op_df
del train_data
gc.collect()

print('user-product')
print(up_df.head())
print('shape:',up_df.shape)

print('unique users in user-product')
up_df['user_id'].nunique()


users = list(set(up_df['user_id']))
print('creating list of users')
user_set1 = users[0:311*1]
user_set2 = users[311*1:311*2]
user_set3 = users[311*2:311*3]
user_set4 = users[311*3:311*4]
user_set5 = users[311*4:311*5]
user_set6 = users[311*5:311*6]
user_set7 = users[311*6:311*7]
user_set8 = users[311*7:311*8]

print('creating subsets for users..')
up_df1 = up_df[up_df['user_id'].isin(user_set1)]
up_df2 = up_df[up_df['user_id'].isin(user_set2)]
up_df3 = up_df[up_df['user_id'].isin(user_set3)]
up_df4 = up_df[up_df['user_id'].isin(user_set4)]
up_df5 = up_df[up_df['user_id'].isin(user_set5)]
up_df6 = up_df[up_df['user_id'].isin(user_set6)]
up_df7 = up_df[up_df['user_id'].isin(user_set7)]
up_df8 = up_df[up_df['user_id'].isin(user_set8)]

del user_set1
del user_set2
del user_set3
del user_set4
del user_set5
del user_set6
del user_set7
del user_set8

# del users
# del up_df
gc.collect()

print('creating subsets..')
up_df1 = pd.crosstab(up_df1.user_id,up_df1.product_pair).astype('uint32')
up_df2 = pd.crosstab(up_df2.user_id,up_df2.product_pair).astype('uint32')
up_df3 = pd.crosstab(up_df3.user_id,up_df3.product_pair).astype('uint32')
up_df4 = pd.crosstab(up_df4.user_id,up_df4.product_pair).astype('uint32')
up_df5 = pd.crosstab(up_df5.user_id,up_df5.product_pair).astype('uint32')
up_df6 = pd.crosstab(up_df6.user_id,up_df6.product_pair).astype('uint32')
up_df7 = pd.crosstab(up_df7.user_id,up_df7.product_pair).astype('uint32')
up_df8 = pd.crosstab(up_df8.user_id,up_df8.product_pair).astype('uint32')

print('merging into one crosstab')
user_prodpair = pd.concat([up_df1,up_df2,up_df3,up_df4,up_df5,up_df6,up_df7,up_df8])

print('user-product pair')
#print(user_prodpair.head())
print(user_prodpair.shape)
#print(user_prodpair['user_id'].nunique())

print('Trimming columns')
col_sums = list(user_prodpair.sum(axis=0))

med = np.median(col_sums)

#Only keeping columns whose sum is greater than median
user_prodpair = user_prodpair[list(user_prodpair.columns[user_prodpair.sum()>med])]
print(user_prodpair.shape)

#Doin a PCA
explained_variance=0
n=5
while explained_variance<0.90:
    print('Doing PCA..')
    pca = PCA(n_components=n)
    pca.fit(user_prodpair)
    pca_samples = pca.transform(user_prodpair)
    
    explained_variance = pca.explained_variance_ratio_
    print('number of components:',n)
    print('explained ratio:',explained_variance)
    n +=1

ps = pd.DataFrame(pca_samples)
print('Principal Components')
print(ps.head())

ps.to_csv('p_components.csv')

