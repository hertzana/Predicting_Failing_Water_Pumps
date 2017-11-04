# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:57:51 2017

@author: hanzhu
"""
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


indep_vars = pd.read_csv('predictor_vars.csv')
outcome = pd.read_csv('outcome_vars.csv')

data_org = pd.merge(indep_vars, outcome, on='id')
#########################################################################################
#######################  Funder & Installer Consolidation ###############################
#########################################################################################

# Consolidate 'funder' and 'installer' into categories

############################## Funder ######################################################
# First, there is a huge number of funders:
len(data['funder'].value_counts().index.unique())
# There are a total of 1,897 funders

# So we'll try to consolidate the funders into the following categories
# 1. Private Company - if the organization appears to be commercial in nature
# 2. Tanzania Government - department/organization that appears to part of Tanzania government
# 3. Other Govt/Govt Body - other national governments (e.g. Germany, Japan) or international governing 
        # bodies (e.g. UN, European Union)
# 4. Nonprofit/NGO/Foundations (e.g. Oxfam, World Bank, Global Fund)
# 5. Religious Organizations (e.g. missionaries, churches)
# 6. Tanzania Organizations - an organization that's not part of the Tanzania Government (i.e. Tanzania-based nonprofits)
# 7. Other - organization that cannot be easily categorized or is unknown
        
# Another reason for consolidation is that there is significant data cleaning needed. Many funder names are misspelled 
# or spelled in different ways, and each unique spelling has its own entry in the data. For example, World Bank can be
# spelled as 'World Bank', 'worldbank', 'wauld banks', etc., and each spelling is considered a different entry in the data.
        
# A variable 'funder_type' will be created to contain the category of funder
# Regex is used to consolidate funder types

# Private companies
comp_list = ["^[Aa].*[G,g]ermany", "Ltd*$", "[C,c]ompa.*(?!sion)$", "Gold Mine", "gmbh", 
            "Priva.*", "Operation", "Fresh Water Plc", "Ikeuchi Towels Japan", '^Pri.*[vae]$', '[Cc]ompany', 'INSTITUTIONS']

# Tanzania government
natlgov_list = ["[D,d]omestic", "[T,t]anzania.*(Gvt|Government)", "Government of Tanzania", 
                "Government/school", "[M,m]inistry", "^Rural", "^Water (Authority|Department|Board|Sector)", 
                "^National"]

# Other national governments / governing bodies
othergov = ['^(Belgi|China|Egypt|Finland|Irish|Italy|Japan).*(Government).*', 
            '^(Canada|Egypt|Finland|France|Frankfurt|Greec|Hol+and|Ital.*|Japan|Ne.*th.*|Norway.*|Swe.*d.*)$',
            '[E,e]mbas+y', 'British Tanza', '^People.*', '^The People', '[E,e]uropean Union', 'Germany Republi.*', 'Irish Ai', 
            'Opec.*', 'Tz As.*', 'Tz Japan.*', 'U.S.A.*', '[U,u]ndp.*', 'Unesco', 'Unhcr.*', 'Usaid.*', 'Swidish']

# Nonprofit/NGO/Foundations 
nonprof = ['Acord', '^Action', 'Afri+ca.*', 'Asb', 'Bingo', 'Bread', 'Busoga', '[A,a]id', '[C,c]are', 'Compasion', 
           '^Conce', 'Cpps', '[D,d]anida', '[D,d]esk.*[C,c]hair', 'Engineers With', '^[F,f]in.*[W,w]ater$', 'Finw',
           'Global Fund', '[F,f]oundation', 'Government /sda', '.*[W,w]orld.*[V,v]ision.*', '.*[W,w]orld.*[B,b]ank.*', 
           '^Imf$', 'Food', '[N,n]go', '[I,i]nternational', '^[O,o]iko.*[sa]$', '^[O,o]x.*', '^[P,p]eace.*', '^[P,p]lan',
           '[R,r]ed.*[C,c]ross$', '^[R,r]ot.*y.*', '^Tanz.*((per)|([O,o]p))$', 'Tanzania /egypt', '^[U,u]nice.*', 
           '^[W,w]ater.*((sema)|(dwe))$', '^[W,w]ater$', '[W,w]fp', 'Farm-africa'] 

# Religious organizations
religious = ['[M,m].*si*l[ie][mus]+', '[C,c]hur[ch]+', '[S,s]ain[ts]', '[C,c].*rist[ia]+[ns]+', '[D,d]ioce[se]*', 
             '[Aa]nglican', '[Mm]ethodist', '[Mm]i+s+i+on+ar', '[Is]lam', 'Neemia Mission', '^Rc.*', '[C,c]ath.*ic', 
             '^[R,r]oman.*', 'Isla$', '[Bb].ptist', '^Missio$']

# Tanzania Organizations
tanz_org = ['[Fr[ie]+dk.*', '[Tt]as+[asfde]+']         
        
# Creater funder type organization        
data['funder_type'] = "Other"        
        
for i in data['funder'].value_counts().index.unique().tolist():
    if re.search("|".join(comp_list), i):
        data.loc[data['funder']==i, 'funder_type'] = "Private Company"
    elif re.search("|".join(natlgov_list), i):
        data.loc[data['funder']==i, 'funder_type'] = "Tanzania Government"
    elif re.search("|".join(othergov), i):
        data.loc[data['funder']==i, 'funder_type'] = "Other Govt/Govt Body"
    elif re.search("|".join(nonprof), i):
        data.loc[data['funder']==i, 'funder_type'] = "Nonprofit/NGO/Foundation"
    elif re.search("|".join(religious), i):
        data.loc[data['funder']==i, 'funder_type'] = "Religious Organization"
    elif re.search("|".join(tanz_org), i):
        data.loc[data['funder']==i, 'funder_type'] = "Tanzania Organization"      

# correct organizations categorized incorrectly by regex         
relabel = {"Other": ['Private Individual', 'Private Individul', 'Private Person', 'Hongoli', 'Chongolo', 'Kiwanda Cha Ngozi',
                     'Majengo Prima', 'Mango Tree', 'Mwalimu  Maneromango Muhenzi', 'African Barrick Gold', 
                     'Namungo Miners', 'Nyabarongo Kegoro', 'Nyamongo Gold Mining', "Oak'zion' And Bugango B' Commu",
                     'Owner Pingo C', 'Ringo','Said Hashim', 'Said Omari', 'Said Salum Ally','Saidi Halfani',
                     'Shamte Said', 'Simango Kihengu','Siza Mayengo', 'St Elizabeth Majengo', 'Total Land Care',
                     'Total Landcare', 'Totaland Care', 'Totoland Care', 'British Tanza'],
            "Religious Organization": ['Haidomu Lutheran Church',  'African Muslim Agency',  "Ju-sarang Church' And Bugango", 
                          'Moslem Foundation',  'Muslimehefen International'], 
            "Nonprofit/NGO/Foundation": ['Compasion International', 'Tanzania Compasion', 'Africa Project Ev Germany']}        
        
for key in relabel:
    for val in relabel[key]:
        data.loc[data['funder']==val, 'funder_type'] = key 

# Number of observations for each funder type         
data['funder_type'].value_counts(dropna=False)        
# Output:
#Other                       41804
#Nonprofit/NGO/Foundation    10813
#Other Govt/Govt Body         2180
#Religious Organization       1686
#Tanzania Organization        1071
#Tanzania Government          1005
#Private Company               841  
      
data.groupby(['funder_type', 'status_group']).size()
#funder_type               status_group           
#Nonprofit/NGO/Foundation  functional                  5777
#                          functional needs repair      787
#                          non functional              4249
#Other                     functional                 21943
#                          functional needs repair     3214
#                          non functional             16647
#Other Govt/Govt Body      functional                  1509
#                          functional needs repair       77
#                          non functional               594
#Private Company           functional                   714
#                          functional needs repair       15
#                          non functional               112
#Religious Organization    functional                  1270
#                          functional needs repair       95
#                          non functional               321
#Tanzania Government       functional                   434
#                          functional needs repair       49
#                          non functional               522
#Tanzania Organization     functional                   612
#                          functional needs repair       80
#                          non functional               379        
#        
# It appears that the majority of water points funded by 'Other Govt/Govt Body', Private Companies,
# Religious Organization, and Tanzania Organizations (most by Tassaf) are functional. Under funding by
# Tanzania Government, there is a large proportion of non-functional water points.

############################## Installer ######################################################
# Similarly, we will categorize 'installer' into categories

len(data['installer'].value_counts().index.unique())
# There are a total of #2,145 installers in the data

data['installer'].value_counts()[0:50]
# There are a large number of installers that have a large count:
#DWE                           17402
#Government                     1825
#RWE                            1206
#Commu                          1060
#DANIDA                         1050
#KKKT                            898
#Hesawa                          840
#0                               777
#TCRS                            707
#Central government              622
#CES                             610
#Community                       553
#DANID                           552
#District Council                551
#HESAWA                          539
#LGA                             408
#World vision                    408
#WEDECO                          397
#TASAF                           396
#District council                392
#Gover                           383
#AMREF                           329
#TWESA                           316

# For these installers, we will not re-categorize them.


# Consolidate into installer groups
data['installer_group'] = "Other"

privateco_installer = ["^[Aa].*[G,g][Ee][Rr][Mm].*[Yy]", "[Ll][Tt][Dd]", "[C,c][Oo][Mm][Pp][Aa][Nn][Yy]*", "[Mm][Ii][Nn][Ee]", "gmbh", 
           "Operation", "Fresh Water Plc", "Ikeuchi Towels Japan", 'construction', 'CONSTRUCT.*','[Cc][oO]$', 
           '[Ee][Nn][Gg][Ii][Nn][Ee].*','.*[Tt][Ee][Cc][Hh].*',
           '[Ll][Oo][Cc][Aa][Ll].*', '[Cc][Oo][Nn][Tt][Rr].*', '[Dd][Rr][Ii][Ll].*', '[Ww]ell.*', 'BUILD.*', 'build.*', 'TRADE.*', 
           '[Tr]rad.*', '[Cc]onsult', '[Ww]orke[rs]+', '[Pp][Rr][Ii][Vv].*[ateyd]', '[Ii]nstitution.*', 'INSTITUTION.*', 'Enterp', 
           'ENTERP']

natlgov_list_Installer = ["[Dd]omestic", "[Tt]anzania (Gvt|Government)", "Government of Tanzania", 
                          "Government/school", "[Mm][Ii][Nn][Ii][[Ss][Tt][Rr][Yy]", "^Rural", 'RURAL', 
                          "^Water (Authority|Department|[Bb]oar[ds]+|Sector)", 
                          "^National", '[Cc][Ee].*[TtRr]+[Aa][Ll]*.*[gG][Oo][Vv].*[Tt]', '\/*[Gg][Oo][Vv][TtEe].*', 
                          '^[Tt][Aa][Nn][Zz]$']

othergov_installer = ['^(Belgi|China|Egypt|Finland|Irish|Ital[iy].*|Japan).*(Government).*', 
            '^(Canada|[Cc][Hh][ii][Nn][Aa]|[Gg][Ee][Rr][Mm].*[Yy]|Egypt|Finland|France|Frankfurt|Greec|[Hh][Oo][Ll]+[Aa][Nn][Dd]|Ital.*|Japan|Ne.*th.*|Norway.*|Swe.*d.*)$', 
            'JAPAN$',  'ITAL[YI]$', 
            '[E,e][Mm][Bb][Aa][Ss]+[Yy]', '^People.*', '^The People', '[E,e]uropean Union', 'Germany Republi.*', 'Irish Ai', 
            'Opec.*', 'Tz As.*', 'Tz Japan.*', 'U.S.A.*', '[U,u]ndp.*', 'Unesco', '[Uu]nhcr.*', 'Usaid.*', 'Swidish', '^[Uu][Nn]$', 
            'UN Habitat', 'UN ONE', 'UNDP', 'UNHCR', 'Canada.*nia$']

nonprof_installer = ['Acord', '^Action', 'Asb', 'Bingo', 'Bread', 'Busoga', '[A,a]id', '[C,c]are', 'Compasion', 
           '^Conce', 'Cpps', '[D,d]esk.*[C,c]hair', 'Development', 'DEVELOPMENT', 'ENGINEERS WITHOUT.*',
           'Global Fund', '[F,f]oundation', 'Government /sda',
           '^Imf$', 'Food', '[N,n]go', '[I,i]nternational', '^[O,o]iko.*[sa]$', '^[P,p]eace.*', '^[P,p]lan',
           '[R,r][Ee[dD].*[C,c][Rr][Oo][Ss]+$', '^[R,r]ot.*y.*', '^Tanz.*((per)|([O,o]p))$', 'Tanzania /egypt', 
           '^[W,w]ater$', '[W,w][Ff][Pp]', 'Farm-africa', 're.*lief', 'RE.*LIEF', 'Africaone'] 

religious_installer = ['[M,m].*si*l[ie][mus]+', '[C,c][Hh][Uu][Rr][CHch]+', '[S,s]ain[ts]', '[C,c].*rist[ia]+[ns]+', '[D,d]ioce[se]*', 
             '[Aa][Nn][Gg][Ll][Ii].*', '[Mm]ethodist', '[Mm][Ii]+[Ss]+[Ii]+[Oo][Nn]+[Aa][Rr].*', '[Is]lam', 'Neemia Mission', 
             '^[Rr][Cc].*', '[C,c]ath.*ic', 
             '^[R,r]oman.*', 'Isla$', '[Bb].ptist', '^[Mm][Ii][Ss]+[Ii][ONon]*$', '[CH]+RIS[TI]+AN', 'Ndanda missions']

for i in data['installer'].value_counts().index.unique().tolist():
    if re.search("|".join(privateco_installer), i):
        data.loc[data['installer']==i, 'installer_group'] = "Private Company"
    elif re.search("|".join(natlgov_list_Installer), i):
        data.loc[data['installer']==i, 'installer_group'] = "Tanzania Government"
    elif re.search("|".join(othergov_installer), i):
        data.loc[data['installer']==i, 'installer_group'] = "Other Govt/Govt Body"
    elif re.search("|".join(nonprof_installer), i):
        data.loc[data['installer']==i, 'installer_group'] = "Nonprofit/NGO/Foundation"
    elif re.search("|".join(religious_installer), i):
        data.loc[data['installer']==i, 'installer_group'] = "Religious Organization"
  

# The below installers have large counts. For these installers, we will just make sure that the name of the same organization is 
# spelled consistently, so that there are not separate line items for each unique spelling.
installer_relabel = {'Hesawa': ['[Hh][Ee][sS][AaEe][Ww][Aa]*'], 
                     'CES': ['^CES$'],
                     'DANIDA': ['[Dd][Aa][Nn][Ii][DdAa]+.*'], 
                     'DWE': ['^DWE.*', "Consultant and DWE"],
                     'Amref': ['AMREF', 'Amref'],
                     "DA": ["^[Dd][Aa]$"],
                     'ACRA': ['[Aa][Cc]+[Rr][Aa]'],
                     "District or Community": ['[Rr][Ee][Gg][Ii][Oo][Nn]', '[Dd]istri.*', '[Cc][Oo][Uu][Nn][Cc][Ii][Ll]', 
                                               "[Cc][Oo][Mm]+[Uu].*", '^[Vv][Ii][Ll]+.*'],
                     'ADRA': ['^Adra', '^ADRA'],
                     'DH': ['^DH$'],
                     'DMDD': ['Dmdd', 'DMDD'],
                     'DW': ['^DW$'],
                     'Fini Water': ['^[F,f][Ii][Nn].*[W,w].*'],
                     'Halimashauli': ['Ha[li]+mashau[rl]i', ],
                     'ISF': ['^[Ii][SsFf]+$'],
                     'Idara ya maji': ['Idara.*', 'IDARA'],
                     'Grumeti': ['G[ur]+um.*', 'G[UR]+UM.*'],
                     'Jica': ['[Jj][EeAa]*.*[Ii][CcKk].*'], 
                     'Kili Water': ['[Kk][Ii][Ll].*[Rr]$'],
                     'KKKT': ['[Kk]{1,3}[Tt].*'],
                     'Kuwait': ['[Kk][Uu][Ww][Aa][Ii][Tt].*'],
                     'Lawatefuka water sup': ['Lawate.*fuka water su.*'],
                     'LGA': ['lga', 'LGA'],   
                     'Magadini-Makiwaru wa': ['Magadini.*Makiwaru wa'],
                     'MUWASA': ['Muwaza', 'MUWSA'], 
                     'MWE': ['^MWE.*'],
                     'Norad': ['[Nn][Oo][Rr].*[AaDd\/]+$'],
                     'Oikos': ['OIKOS', 'Oikos'],
                     'Oxfam': ['^[Oo][Xx].*'],
                     'RWE': ['^RWE.*'],
                     'Sengerema Water Department': ['[Ss]engerema.*'],
                     'SHIPO': ['^[Ss][Hh][Ii][Pp].*'],
                     'TASAF': ['[Tt][Aa][Ss]+[asfdeASFDE]+'],
                     'TCRS': ['^TCRS.*'],
                     'TWESA': ['^TWESA.*'],
                     'UNICEF': ['^[U,u][Nn][Ii][SsCc][erER].*'],
                     'Wanan': ['^[Ww]anan'],
                     'World Vision': ['[W,w][Oo].*[Dd].*[Di]*[Vv][Ii][Ss].*[Nn]+$'], 
                     'World Bank': ['[W,w][Oo].*[Dd].*[Bb].*[KkSs]+$', '[Ww][Oo].*[Dd]$'],
                     'Water Aid': ['[Ww][Aa][Tt][Ee][Rr].*[Aa][Ii][Dd]'],
                     'SEMA': ['SEMA', 'sema'],
                     'WEDECO': ['[Ww][eE]*[Dd][eE][CcKk][Oo].*'],
                     'WU': ['^WU$'],
                     'WVT': ['^WVT$'],
                     'Other': ['Dokta Mwandulami', 'Total.*[Cc]are$', 'Simango Kihengu', 'Said$', '^Said', 'Nyabarongo Kegoro', 
                               'Yoroko mwalongo', 'Kiwanda cha Ngozi', 'Mpango wa Mwisa', "Ng'omango", 'Namungo', 'Zuber Mihungo']
                     }
       

for key in installer_relabel:
    for i in data['installer'].value_counts().index.unique().tolist():
        if re.search("|".join(installer_relabel[key]), i):
            data.loc[data['installer']==i, 'installer_group'] = key

data['installer_group'].value_counts(dropna=False) 
#DWE                           17432
#Other                         13461
#Tanzania Government            3853
#District or Community          3794
#DANIDA                         1677
#Private Company                1509
#Hesawa                         1402
#RWE                            1292
#KKKT                           1208
#Religious Organization         1121
#Fini Water                      784
#TCRS                            734
#World Vision                    715
#CES                             610
#TASAF                           514
#Jica                            445
#Amref                           443
#LGA                             408
#WEDECO                          407
#Norad                           387
#Nonprofit/NGO/Foundation        386
#DMDD                            377
#UNICEF                          333
#Oxfam                           331
#TWESA                           327
#SEMA                            325
#ACRA                            308
#DA                              308
#World Bank                      304
#WU                              301
#ISF                             291
#Kili Water                      280
#ADRA                            276
#DW                              246
#SHIPO                           236
#Idara ya maji                   232
#Sengerema Water Department      218
#Water Aid                       214
#Wanan                           209
#DH                              202
#MWE                             197
#Kuwait                          196
#Lawatefuka water sup            187
#Magadini-Makiwaru wa            179
#Other Govt/Govt Body            171
#WVT                             158
#Halimashauli                    153
#MUWASA                          108
#Oikos                           101
#Grumeti                          50
        
# Let's reduce the number of categories even further to keep it within 30
data['installer_final'] = data['installer_group']
data.loc[data['installer_group']=='DW', 'installer_final']="Other"          
data.loc[data['installer_group']=='SHIPO', 'installer_final']="Other"          
data.loc[data['installer_group']=='Idara ya maji', 'installer_final']="Other"          
data.loc[data['installer_group']=='Sengerema Water Department', 'installer_final']="District or Community"          
data.loc[data['installer_group']=='Water Aid', 'installer_final']="Nonprofit/NGO/Foundation"        
data.loc[data['installer_group']=='Wanan', 'installer_final']="Other"
data.loc[data['installer_group']=='DH', 'installer_final']="Other"        
data.loc[data['installer_group']=='MWE', 'installer_final']="Other"        
data.loc[data['installer_group']=='Kuwait', 'installer_final']="Other Govt/Govt Body"        
data.loc[data['installer_group']=='Lawatefuka water sup', 'installer_final']="Other"        
data.loc[data['installer_group']=='Magadini-Makiwaru wa', 'installer_final']="Other"        
data.loc[data['installer_group']=='WVT', 'installer_final']="Other"        
data.loc[data['installer_group']=='Halimashauli', 'installer_final']="Other"        
data.loc[data['installer_group']=='Oikos', 'installer_final']="Nonprofit/NGO/Foundation"       
data.loc[data['installer_group']=='Grumeti', 'installer_final']="Other"          
data.loc[data['installer_group']=='MUWASA', 'installer_final']="Other"            


data['installer_final'].value_counts(dropna=False) 
# Final Output
#DWE                         17432
#NaN                         13242
#District or Community        4012
#Tanzania Government          3981
#Other                        2206
#DANIDA                       1677
#Private Company              1518
#Hesawa                       1402
#RWE                          1292
#KKKT                         1208
#Religious Organization       1130
#Fini Water                    784
#TCRS                          734
#World Vision                  715
#Nonprofit/NGO/Foundation      701
#CES                           610
#TASAF                         514
#Jica                          445
#Amref                         443
#LGA                           408
#WEDECO                        407
#Other Govt/Govt Body          391
#Norad                         387
#DMDD                          377
#UNICEF                        333
#Oxfam                         331
#TWESA                         327
#SEMA                          325
#DA                            308
#ACRA                          308
#World Bank                    304
#WU                            301
#ISF                           291
#Kili Water                    280
#ADRA                          276

data.groupby(['installer_final', 'status_group']).size()
#installer_final           status_group           
#ACRA                      functional                  302
#                          non functional                6
#ADRA                      functional                  126
#                          functional needs repair       2
#                          non functional              148
#Amref                     functional                  189
#                          functional needs repair       6
#                          non functional              248
#CES                       functional                  538
#                          functional needs repair       1
#                          non functional               71
#DA                        functional                  216
#                          non functional               92
#DANIDA                    functional                 1039
#                          functional needs repair      95
#                          non functional              543
#DMDD                      functional                  344
#                          functional needs repair       7
#                          non functional               26
#DWE                       functional                 9455
#                          functional needs repair    1622
#                          non functional             6355
#District or Community     functional                 2233
#                          functional needs repair     231
#                          non functional             1548
#Fini Water                functional                  124
#                          functional needs repair      33
#                          non functional              627
#Hesawa                    functional                  791
#                          functional needs repair      54
#                          non functional              557
#ISF                       functional                  201
#                          functional needs repair      16
#                          non functional               74
#Jica                      functional                  312
#                          functional needs repair      21
#                          non functional              112
#KKKT                      functional                  593
#                          functional needs repair      78
#                          non functional              537
#Kili Water                functional                  166
#                          functional needs repair       3
#                          non functional              111
#LGA                       functional                  102
#                          functional needs repair      81
#                          non functional              225
#Nonprofit/NGO/Foundation  functional                  474
#                          functional needs repair      39
#                          non functional              188
#Norad                     functional                  163
#                          functional needs repair      48
#                          non functional              176
#Other                     functional                 1400
#                          functional needs repair      83
#                          non functional              723
#Other Govt/Govt Body      functional                  184
#                          functional needs repair      23
#                          non functional              184
#Oxfam                     functional                  178
#                          functional needs repair      65
#                          non functional               88
#Private Company           functional                  922
#                          functional needs repair      66
#                          non functional              530
#RWE                       functional                  334
#                          functional needs repair     140
#                          non functional              818
#Religious Organization    functional                  786
#                          functional needs repair      63
#                          non functional              281
#SEMA                      functional                  227
#                          functional needs repair      17
#                          non functional               81
#TASAF                     functional                  310
#                          functional needs repair      37
#                          non functional              167
#TCRS                      functional                  305
#                          functional needs repair      43
#                          non functional              386
#TWESA                     functional                  207
#                          functional needs repair      37
#                          non functional               83
#Tanzania Government       functional                 1310
#                          functional needs repair     340
#                          non functional             2331
#UNICEF                    functional                  206
#                          functional needs repair      12
#                          non functional              115
#WEDECO                    functional                  278
#                          functional needs repair      18
#                          non functional              111
#WU                        functional                  261
#                          non functional               40
#World Bank                functional                  126
#                          functional needs repair      26
#                          non functional              152
#World Vision              functional                  439
#                          functional needs repair      89
#                          non functional              187


# Installers for which large majority of wells are functional: ACRA, CES, DA, Danida, DH, DMDD, 
# District/Community, ISF, Jica, Nonprofits/NGOs, Other, Oxfam, Private Company,Religious Organization,
# SEMA, Tasaf, TWESA, WEDECO, WU, World Vision,  Unicef

# Installer for which most wells are functional, but also have large number of non-functional wells:
# DWE, Hesawa
 
# Installers fow which majority of wells are not functional: Amref, Fini Water, LGA, RWE, TCRS, Tanzania Governmnt,
# World Bank 

# Save dataset with 'installer_group' and 'funder_type'
data.to_csv('water2.csv')

