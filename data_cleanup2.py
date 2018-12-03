# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:58:51 2017

@author: hanzhu
"""

#########################################################################################
####################### Data Cleaning ###################################################
#########################################################################################

########################################################################################
# In this file we clean other variables in the data: we resolve missing values, conduct
# data normalization and standardization, construct additional useful
# features, and drop unneeded variables
########################################################################################

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
import os
import math
import re
from scipy import stats
from scipy.interpolate import interp1d
from sklearn import preprocessing
os.chdir("C:\\Users\\hanzhu\\Documents\\DAT210x-master\\A - Water")

# Read in previous data with installer and funder consolidated into categories
data = pd.read_csv('water2.csv')

#### Longitude ####################################################
# Missing longitude have values '0'
# Fill in missing values with average longitude - as all water points are in Tanzania, 
# we can use the average.

min(data[data['longitude']!=0]['longitude']) #29.6
max(data['longitude']) # 40.345
min(data['longitude']) # 40.345

#Mean of longitude
np.mean(data[data['longitude']!=0]['longitude']) 
np.median(data[data['longitude']!=0]['longitude'])
# Mean and mdian both around 35, so take the mean

data.loc[data['longitude']==0, 'longitude'] = np.mean(data[data['longitude']!=0]['longitude']) 

#### Latitude ####################################################
# Missing latitudes have values -2e-08, very close to 0. The next smallest value is approx. 1
min(data['latitude']) # -11.649440179999999
max(data['latitude']) # -2e-08
max(data[data['latitude']!=-2e-08]['latitude']) #$-0.999

sorted(data['latitude'].value_counts().index.unique())
len(data[data['latitude']==-2e-08]) #1,812

np.mean(data[data['latitude']!=-2e-08]['latitude'])
np.median(data[data['latitude']!=-2e-08]['latitude'])

data.loc[data['latitude']==-2e-08, 'latitude'] = np.mean(data[data['latitude']!=-2e-08]['latitude'])

#### Region ####################################################
# Region codes are not consistent with region
# Arusha = 2, 24
# Lindi = 80, 8, 18; but 18 is either Kagera or Lindi. Lindi sometimes incorrectly coded as 18
# Mtwara = 90, 99, 9; inconsistent coding
# Pwani = 6, 60, 40
# shinyanga = 17, 14, 6

# for each region, check what the corresponding region codes are
data[data['region']=='Tanga']['region_code'].value_counts()

data[data['region_code']==5]['region'].value_counts()

# Codes are inconsistent, so we will use region instead and drop region code

# What does the distribution look like
data['region'].value_counts()

#### LGA ####################################################

# Appears LGA corresponds to districts

len(data['lga'].value_counts(dropna=False).index.unique().tolist()) #125 LGA's
len(data['lga'].value_counts(dropna=False)) #

LGA_flag = {}

# Verify accuracy of LGA; are there any LGA's that correspond to more than one region?
for i in data['lga'].value_counts().index.unique():
    if len(data[data['lga']==i]['region'].value_counts().index.unique())>1:
        LGA_flag[i] = data[data['lga']==i]['region'].value_counts().index.unique().tolist()

# returns empty string, so each LGA corresponds to only 1 region   

# What does the distribution look like?
data['lga'].value_counts()

#### Public Meeting ####################################################

# How many missing values?
data['public_meeting'].value_counts(dropna=False)  #3334 missing values      
        
#### Scheme Management ####################################################
data['scheme_management'].value_counts(dropna=False)   
#VWC                 36793
#WUG                  5206
#NaN                  3877
#Water authority      3153
#WUA                  2883
#Water Board          2748
#Parastatal           1680
#Private operator     1063
#Company              1061
#Other                 766
#SWC                    97
#Trust                  72
#None                    1    
data['scheme_name'].value_counts(dropna=False) # many different scheme names

data['management'].value_counts(dropna=False) 
# Take this as source of truth for scheme management, as this is a more consolidated version of 
# scheme_management and doesn't have 'None'
#vwc                 40507
#wug                  6515
#water board          2933
#wua                  2535
#private operator     1971
#parastatal           1768
#water authority       904
#other                 844
#company               685
#unknown               561
#other - school         99
#trust                  78
data['management_group'].value_counts(dropna=False)
#user-group    52490
#commercial     3638
#parastatal     1768
#other           943
#unknown         561


#### Extraction Type ####################################################
data['extraction_type'].value_counts(dropna=False)
data['extraction_type_group'].value_counts(dropna=False)
# Group has larger sizes, so drop extraction type

#### Payment Type ####################################################
data['payment'].value_counts(dropna=False)
data['payment_type'].value_counts(dropna=False)
# Exactly the same variables, keep payment_type

#### Water Quality ####################################################
data['water_quality'].value_counts(dropna=False)
#soft                  50818
#salty                  4856
#unknown                1876
#milky                   804
#coloured                490
#salty abandoned         339
#fluoride                200
#fluoride abandoned       17
data['quality_group'].value_counts(dropna=False)
#good        50818
#salty        5195
#unknown      1876
#milky         804
#colored       490
#fluoride      217
# Just take quality group for larger sample size of each category

#### Water Quantity ####################################################
data['quantity'].value_counts(dropna=False)
data['quantity_group'].value_counts(dropna=False)
# exactly same, drop quantity group

#### Source ####################################################
data['source'].value_counts(dropna=False)
data['source_type'].value_counts(dropna=False)
data['source_class'].value_counts(dropna=False)
# take source type, but also keep source class

#### Water Point Type ####################################################
data['waterpoint_type'].value_counts(dropna=False)
data['waterpoint_type_group'].value_counts(dropna=False)

# Put dam category into 'other', as sample size is just 7 
data.loc[data['waterpoint_type_group']=='dam', 'waterpoint_type_group'] = 'other'


#### Date Recorded ####################################################
# Make 2 new vars:
# 1. Extract year from date recorded to create 'year recorded'
# 2. Extract month-year --> there are many records made in March 2011, for example, so we want to 
#                           see if the month-year together has an effect on the model

# Check for missing values
data['date_recorded'].value_counts(dropna=False)

data['date_recorded'] = pd.to_datetime(data['date_recorded'])

data['year'] = data['date_recorded'].map(lambda x: x.year)

data['month-year'] = data['date_recorded'].map(lambda x: str(x.year) +"-"+str(x.month))

other = ['2011-1$', '2011-6', '2011-10', '2011-9', '2011-5', '2012-1$', '^2004', '^2002']

for i in data['month-year'].value_counts().index:
    if re.search("|".join(other), i):
        data.loc[data['month-year']==i, 'month-year'] = 'other'


data['month-year'].value_counts()

#### Construction Year ####################################################

data['construction_year'].value_counts(dropna=False)
# About 1/3 of data has missing construction years: 20,709 records. Not really a easy way to 
# interpolate year. We can try running model once without construction year, then a second time
# with construction year but remove the rows with missing years. 

np.mean(data[data['construction_year']!=0]['construction_year'])
# Average construction year is 1997


################ Drop variables not needed #############################################################
data = data.drop(['wpt_name', 'subvillage', 'region_code', 'district_code', 'recorded_by', 'scheme_management',
                  'scheme_name', 'extraction_type', 'payment', 'water_quality', 'quantity_group', 'source', 
                  'waterpoint_type_group', 'installer_group', 'installer', 'funder', 'population_fake',
                  'construction_year_interp', 'date_recorded', 'Unnamed: 0'], axis=1)

# Categorize as other for categories with very few values
data[data['waterpoint_type']=='dam'] = 'other'
data.loc[data['year']==2004, 'year'] = 'other'
data.loc[data['year']==2002, 'year'] = 'other'

#########################################################################################
#######################  Normalization / Standardization  ###############################
#########################################################################################

# Part I: Normalization
#################### amount_tsh ########################
# Look at distribution of numerical variables:
# convert numerical variables coded as strings to numerical
data['amount_tsh'] = pd.to_numeric(data['amount_tsh'])

plt.figure(figsize=(5,5)) 
plt.boxplot(data['amount_tsh'])
plt.show()

stats.skew(data['amount_tsh']) # skew of original amount_tsh = 45
a
a = np.log1p(data['amount_tsh']) # Taking log brings skew down to 1.24
stats.skew(a) # skew = 1.24
plt.figure(figsize=(5,5)) 
plt.hist(a)
plt.show()

b = np.sqrt(data['amount_tsh'])
stats.skew(b)
plt.figure(figsize=(5,5)) 
plt.hist(b)
plt.show()

data['amount_tsh'] = np.log1p(data['amount_tsh'])

# What are the min and max of the transformed 'amount_tsh'?
min(data['amount_tsh']) #0
max(data['amount_tsh']) #12.4
#################### gps height ########################
data['gps_height'] = pd.to_numeric(data['gps_height'])
stats.skew(data['gps_height']) #skew = 0.511

plt.figure(figsize=(5,5)) 
plt.hist(data['gps_height'])
plt.show()

# No need to normalize as skew is already low at 0.511
min(data['gps_height']) #-90
max(data['gps_height']) #2770
#################### longitude ########################
data['longitude'] = pd.to_numeric(data['longitude'])
stats.skew(data['longitude']) #skew = -0.19

plt.figure(figsize=(5,5)) 
plt.hist(data['longitude'])
plt.show()
# No need to normalize as skew is already low at -0.19

min(data['longitude']) #29.6
max(data['longitude']) #40.3
#################### latitude ########################
data['latitude'] = pd.to_numeric(data['latitude'])
stats.skew(data['latitude']) #skew = -0.26

plt.figure(figsize=(5,5)) 
plt.hist(data['latitude'])
plt.show()
# No need to normalize

min(data['latitude']) #-11.6
max(data['latitude']) #-1

#################### population ########################
data['population'] = pd.to_numeric(data['population'])
stats.skew(data['population']) #skew = 13

plt.figure(figsize=(5,5)) 
plt.hist(data['latitude'])
plt.show()

a = np.log1p(data['population'])
stats.skew(a) #0.13

b = np.sqrt(data['population'])
stats.skew(b) #1.94

data['population'] = np.log1p(data['population'])

min(data['population']) #0
max(data['population']) #10.3



# Part II: Standardization
#################### Apply Robust Scaler ########################
# Robust scaler will be used for 'gps height', as there are large outliers
scaler = preprocessing.RobustScaler()

varsdf = pd.DataFrame(data[['amount_tsh', 'gps_height']])

z = scaler.fit_transform(varsdf)

amount_tsh_rob = []
gps_height = []
for x in range(len(z)):
    amount_tsh_rob.append(z[x][0])
    gps_height.append(z[x][1])

amount_tsh_rob = pd.Series(amount_tsh_rob)
gps_height = pd.Series(gps_height)

plt.scatter(amount_tsh_rob, gps_height)
plt.scatter(data['amount_tsh'], data['gps_height'])


# Replace gps_height and amount_tsh with the scaled values
# First, reset indices from 1 to len(data) after dropping vars, so that we no longer have indices
# running to the full length of the original data, with the indices of the dropped rows
# omitted.
# This will be needed when we merge in variables rescaled by robust scaler into the dataset. 
# Otherwise, merging will introduce NA values due to indices not matching as a result of 
# dropped rows from data.

# first store original index
index_orig = data.index # type index
datacopy = data.copy()
# Then reindex
data = data.reset_index(drop=True)

data['amount_tsh_rob'] = amount_tsh_rob
data['gps_height_rob'] = gps_height

# Keep copy of cleaned data
data.to_csv('water3.csv')