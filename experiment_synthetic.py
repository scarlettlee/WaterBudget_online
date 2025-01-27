# %matplotlib inline
from matplotlib.pyplot import figure
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange, sin, cos, pi, power, arctan, abs
import re
from math import comb
from itertools import combinations
import itertools
from warnings import simplefilter
import matplotlib as mpl
from globVar import compute_stats

mpl.rcParams['axes.linewidth'] = 1.5 #set the value globally
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
simplefilter(action="ignore", category=FutureWarning)

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

# generate sum of two sine waves
def getSumSinParams(A_,a_,B_,b_):
    pSin = A_*sin(a_)+B_*sin(b_)
    pCos = A_*cos(a_)+B_*cos(b_)
    sA = power(power(pCos,2)+power(pSin,2),0.5)
    sB = arctan(pSin/pCos)
    return [sA, sB]

# initiate parameters
w = pi/6
# # ET
# A = 3
# # R
# B = 1
# # S
# C = 3
# a = 6
# b = 4
# c = 2
# P
A = 3
# ET
B = 2.5
# R
C = 1
a = 5
b = 4
c = -1

# get two sines: A+B
sumParam0 = getSumSinParams(A,a,-B,b)
am0 = sumParam0[0]
bp0 = sumParam0[1]
# print("sum of A and B\n",sumParam0,am0,bp0)

# get three sines: A+B+c
sumParam = getSumSinParams(am0,bp0,-C,c)
am = sumParam[0]
bp = sumParam[1]
print("sum of A and B and C\n",sumParam,am,bp)

# True values
t = np.linspace(0, 32*np.pi, 100)
# P = am*sin(w*t+bp) + 3
# ET = A*sin(w*t+a)  + 3
# R = B*sin(w*t+b)  + 1
# S = C*sin(w*t+c)  - 1
P = A*sin(w*t+a)  + 3.5
ET = B*sin(w*t+b)  + 3
R = C*sin(w*t+c)  + 1.5
S = am*sin(w*t+bp) - 1
# res = P-ET-R-S
# print(res)

###### generate observations ##################################################################
np.random.seed(0)
# 0 is the mean of the normal distribution you are choosing from
# 1 is the standard deviation of the normal distribution
# 100 is the number of elements you get in array noise
noise = np.random.normal(0,0.1,100)
P1 = P+noise
# np.random.seed(2)
noise = np.random.normal(0,0.2,100)
P3 = P+noise
noise = np.random.normal(0,0.3,100)
P4 = P+noise
noise = np.random.normal(0,0.4,100)
P5 = P+noise
np.random.seed(6)
noise = np.random.normal(0,0.1,100)
P2 = P+noise
# Artificial system error
P2[58] = 20

np.random.seed(1)
noise = np.random.normal(0,0.1,100)
ET1 = ET+noise
noise = np.random.normal(0,0.2,100)
ET2 = ET+noise
noise = np.random.normal(0,0.3,100)
ET3 = ET+noise

np.random.seed(2)
noise = np.random.normal(0,0.3,100)
R1 = R+noise
noise = np.random.normal(0,0.2,100)
R2 = R+noise
noise = np.random.normal(0,0.1,100)
R3 = R+noise

np.random.seed(3)
noise = np.random.normal(0,0.2,100)
S1 = S+noise
noise = np.random.normal(0,0.3,100)
S2 = S+noise
noise = np.random.normal(0,0.4,100)
S3 = S+noise

noise1 = np.random.normal(0,0.1,100)
noise2 = np.random.normal(0,0.2,100)
noise3 = np.random.normal(0,0.3,100)
noise4 = np.random.normal(0,0.4,100)

##### sine visualization ################################################################
# fig = figure(1)

# ax1 = fig.add_subplot(711)
# ax1.plot(t, A*sin(w*t+a))
# ax1.title.set_text('A')

# ax2 = fig.add_subplot(712)
# ax2.plot(t, B*sin(w*t+b))
# ax2.title.set_text('B')

# ax3 = fig.add_subplot(713)
# ax3.plot(t, C*sin(w*t+c))
# ax3.title.set_text('C')

# ax4 = fig.add_subplot(714)
# ax4.plot(t, am0*sin(w*t+bp0))
# ax4.title.set_text('equation derived sum(A,B)')

# ax5 = fig.add_subplot(715)
# ax5.plot(t, am*sin(w*t+bp))
# ax5.title.set_text('equation derived sum(A,B,C)')

# ax6 = fig.add_subplot(716)
# ax6.plot(t, A*sin(w*t+a)+B*sin(w*t+b))
# ax6.title.set_text('A + B')

# ax7 = fig.add_subplot(717)
# ax7.plot(t, A*sin(w*t+a)+B*sin(w*t+b)+C*sin(w*t+c))
# ax7.title.set_text('A + B + C')

# # fig.tight_layout()
# plt.show()


# plt.plot(t,P1,'--',label ='P2', alpha=.6)
# plt.plot(t,P1,'--',label ='P3', alpha=.6)
# plt.plot(t,P1,'--',label ='P4', alpha=.6)
# plt.plot(t,P1,'--',label ='P5', alpha=.6)

plt.plot(t,P,label ='P')
plt.plot(t,ET,label ='ET')
plt.plot(t,R,label ='R')
plt.plot(t,S,label ='TWSC')

plt.plot(t,noise1,'--',label ='ns 0.1')
plt.plot(t,noise2,'--',label ='ns 0.2')
plt.plot(t,noise3,'--',label ='ns 0.3')
plt.plot(t,noise4,'--',label ='ns 0.4')

plt.plot(t,P-ET-R-S,label = r'$\Delta Res$', linewidth=2)

plt.xlabel('month')
plt.ylabel('value')
# plt.legend(loc='lower right', bbox_to_anchor=(0,0))
plt.legend(framealpha=1)
plt.show()

###### validation ###############################################################################################
# universal variables and functions
introObs = True
test = False
dict ={
'P1':P1,
'P2':P2,
'P3':P3,
'P4':P4,
'P5':P5,
'P_closed':P,
'E1':ET1,
'E2':ET2,
'E3':ET3,
'E_closed':ET,
'R1':R1,
'R2':R2,
'R3':R3,
'R_closed':R,
'S1':S1,
'S2':S2,
'S3':S3,
'S_closed':S
}
data = pd.DataFrame(dict)
# print(data)

if test:
    data = data.head(6)
else:
    data = data

# columns and combination
col = data.columns
# stat
df_stat = pd.Series()
lab = ['P', 'E', 'R', 'S']
met = ['PR', 'CKF', 'MCL', 'MSD']
statistics = ['PBIAS','CC','RMSE','ME','ME1','MAE','MAPE']
outputDir = os.path.join(os.path.dirname(__file__), '', 'tmp/')
data.to_csv(outputDir+'oriData.csv')

# get component num order
def getNumOrder(m):
    r = re.compile(m + "[12345](?!\d)$")
    filtered = list(filter(r.match, col))

    e = [i[1] for i in filtered]
    return e

# compute merged true values
def getMergedTrue(arr):
    m = arr.mean()
    sig = arr - m
    t = np.power(sig, -2, dtype=float)
    s = t.sum()
    w = t / s
    return (arr * w).sum()
# def getMergedTrue(arr):
#     m = arr.mean()
#     return m

# compute stats
def computePBIAS(trueValues, correctedValues):
    return 100*((trueValues-correctedValues).sum())/trueValues.abs().sum()

def computeCC(trueValues, correctedValues):
    return trueValues.corr(correctedValues)

def computeCCinv(trueValues, correctedValues):
    return 1-trueValues.corr(correctedValues)

def computeRMSE(trueValues, correctedValues):
    return ((trueValues - correctedValues) ** 2).mean() ** .5

def computeME(trueValues, correctedValues):
    return (trueValues-correctedValues).mean()

def computeME1(trueValues, correctedValues):
    return (trueValues-correctedValues.mean()).abs().mean()

def computeMAE(trueValues, correctedValues):
    return (trueValues-correctedValues).abs().mean()

def computeMAPE(trueValues, correctedValues):
    return 100*((trueValues-correctedValues)/trueValues).abs().mean()

# generate coefficient matrix for MCL
# 1/(2*(n-1)C2) * theta_jk
def generateAInverse(paramNum):
    denominator = 2*comb(paramNum-1,2)
    
    myrows = np.arange(1,paramNum+1)
    A = []    
    # iterate N rows
    for N in myrows:           
        B = []
        # iterate combinations for columns
        combin = combinations(myrows,2)
        for c in combin:
            j = c[0]
            k = c[1]

            if N == j:
                B += [paramNum-2]
            elif N == k:
                B += [paramNum-2]
            elif (N != j) & (N != k):
                B += [-1]
        
        A += [[x / denominator for x in B]]

    return A

# compute squared distance between two series
def getSquaredDistance(x,y):
    # here we are computing every thing
    # step by step
    p1 = np.sum([(a * a) for a in x])
    p2 = np.sum([(b * b) for b in y])
    
    # using zip() function to create an
    # iterator which aggregates elements 
    # from two or more iterables
    p3 = -1 * np.sum([(2 * a*b) for (a, b) in zip(x, y)])
    size = x.size

    return np.sum(p1 + p2 + p3)/size

# generate combinations
Combinations = [] # ['1111', '1112'...]
for e1,e2,e3,e4 in itertools.product(getNumOrder('P'), getNumOrder('E'), getNumOrder('R'), getNumOrder('S')):
    Combinations += [e1+e2+e3+e4]   
# print('combinations\n',Combinations)   

# match remote sensing observations
r = re.compile("[PERS]\d")
colFiltered = list(filter(r.match, col))  # ['P1', 'E1', 'R1'...]

# give data true columns: P/E/R/S
for m in lab:
    data[m] = -9999.0

# give data D columns: P/E/R/S#_D
for colF in colFiltered:
    data[colF + '_D'] = -9999.0
# print('data\n',data.columns)

# give data w columns: PR/CKF/MCL/MSD_####_P/E/R/S_w
# give data corrected value columns: MSD_####_P/E/R/S
for combin in Combinations:
    for m in lab:
        for k in met:
            data[k + '_' + combin + '_' + m] = -9999.0
            data[k + '_' + combin + '_' + m + '_w'] = -9999.0
    
###### the first step #########################################################################################
# What is the best estimation product?
# Validate the best selection through P1-P5 [5333]
# Compute "true" values
for index, row in data.iterrows():
    for m in lab:
        # introduce observation
        if m == 'P' and introObs:
            data[m][index] = data[m+'_closed'][index]
            continue

        # raw
        r = re.compile(m + "[12345](?!\d)$")
        filtered = list(filter(r.match, col))

        arr = data[filtered].iloc[index].to_numpy()        
        data[m][index] = getMergedTrue(arr)
    
# remote sensing data
# Compute pbias, cc, rse, me, mae, mape for remote sensing data
# Traverse P/E/S loop
for m in lab: #["P"]:
    r = re.compile(m+"\d")
    rsFiltered = list(filter(r.match, colFiltered))  # ['P1', 'P2', 'P3'...] ['E1','E2'...]
    
    # Compute pbias, cc, rse for remote sensing
    individualStat = pd.Series()
    
    # Traverse for computing P1_PBIAS
    for o in rsFiltered: # ['P1'] ['P2']            
        t = data[m+'_closed'][3:]
        c = data[o][3:]

        # Traverse stats
        for stat in ['PBIAS','CCinv','RMSE']:
            s = globals()['compute'+stat](t,c)
            individualStat[o+'_'+stat] = s  # P1_PBIAS, P1_invCC, P1_RMSE
    # print("## P1_PBIAS ###########")
    # print(individualStat)

    # Traverse for computing P1_PBIAS_rank
    ind = individualStat.index.values
    for stat in ['PBIAS','CCinv','RMSE']:
        r = re.compile(".*"+stat)
        filtered4rank = list(filter(r.match, ind))
        
        ranked = individualStat[filtered4rank].rank().add_suffix('_rank')
        individualStat = pd.concat([individualStat,ranked]) # P1_PBIAS_rank
    # print("## P1_PBIAS_rank ######")
    # print(individualStat)
    
    # Traverse for computing P1_sum
    ind = individualStat.index.values
    for o in rsFiltered:
        # for stat in ['PBIAS','invCC','RMSE']:
        r = re.compile(o+".*_rank")
        filtered4rank = list(filter(r.match, ind))

        individualStat[o+'_sum'] = individualStat[filtered4rank].sum() # P1_sum
    # print("## P1_sum ######")
    # print(individualStat)        

    # Traverse for computing P1_sumRank        
    ind = individualStat.index.values
    r = re.compile(".*_sum")
    filtered4rank = list(filter(r.match, ind))
    individualStat = pd.concat([individualStat,individualStat[filtered4rank].rank().add_suffix('Rank')])
    # print("## P1_sumRank ######")
    # print(individualStat) 

    df_stat = pd.concat([df_stat,individualStat]) 

# print("The best product (composition) is:\n",df_stat)  
# rank array
combBest = ''
for o in lab:
    r = re.compile(o+'.*_sumRank')
    filtered = list(filter(r.match, df_stat.index.values))

    combBest = combBest + df_stat.loc[filtered].idxmin()[1]
print("best comb:", combBest)
#     df_stat.to_csv(output_dir+fileName[:-4]+'_rank.csv',index=True,header=['values'])

###### the second step ############################################################################
# What is the best BCC method for water budget?
# comupte BCC results

# Construct dictionary for  [PES]_d#t 
# e.g. P has 5 distances, one for each product, i.e. P_d[1-5]t
# distances of all observations
MCL_distances = {}  # 'P2':3
for m in lab:
    r = re.compile(m+"\d")
    param = list(filter(r.match, colFiltered))
    paramNum = len(param)
    # print('-------------------------param',param)

    # A inverse
    A_1 = generateAInverse(paramNum)
    # print("A_1\n",np.array(A_1,dtype=object))
    # iterate combinations for columns
    djk = []
    combin = combinations(np.arange(1,paramNum+1),2)
    for c in combin:
        j = c[0]-1
        k = c[1]-1       

        djk += [getSquaredDistance(data[param[j]],data[param[k]])]
    
    djk = np.array(djk)[np.newaxis]
    # print("djk\n",djk.T)
    y = np.dot(np.array(A_1,dtype=object), djk.T)
    # print('y\n',y)

    # get root and convert lists to dictionary
    res = {param[i]: np.power(abs(y[i][0]),0.5) for i in range(len(param))}
    # print('res\n',res)

    MCL_distances = MCL_distances|res
print('MCL_distances\n',MCL_distances)

################################################################################################################
# iterate rows to compute D, w, and corrected values
first = True
for index, row in data.iterrows():
    print("------------In row: "+str(index)+"------------------------------------------------")
    ######## PR #############################################################################################
    # compute corrected values: PR_####_P/E/R/S
    # lab = ['P', 'E', 'R', 'S']
    # met = ['PR', 'CKF', 'MCL', 'MSD']
    for combin in Combinations: # combination numbers ####
        # get observations
        params = []  # values of the combination P,E,R,S
        for i in [0, 1, 2, 3]:
            params.append(data[lab[i] + combin[i]][index])

        residential = params[0] - params[1] - params[2] - params[3]
        # compute weights w and PR corrected values        
        for i in [0, 1, 2, 3]:
            w = abs(params[i]) / sum([abs(ele) for ele in params])

            data['PR_' + combin + '_' + lab[i] + '_w'][index] = w
            if i == 0:
                w = -w
            data['PR_' + combin + '_' + lab[i]][index] = data[lab[i] + combin[i]][index] + residential * w

    ######## CKF #############################################################################################
    # compute corrected values: CKF_####_P/E/R/S
    # lab = ['P', 'E', 'R', 'S']
    # met = ['PR', 'CKF', 'MCL', 'MSD']
    for combin in Combinations: 
        # get observations
        d = []       # distances of the combination
        params = []  # values of the combination P,E,R,S
        for i in [0, 1, 2, 3]:
            dis = np.power(data[lab[i] + combin[i]][index]-data[lab[i]][index],2)
            d.append(dis)
            params.append(data[lab[i] + combin[i]][index])

        residential = params[0] - params[1] - params[2] - params[3]
        # compute weights w and CKF corrected values        
        for i in [0, 1, 2, 3]:
            w = d[i] / sum(d)

            data['CKF_' + combin + '_' + lab[i] + '_w'][index] = w
            if i == 0:
                w = -w
            data['CKF_' + combin + '_' + lab[i]][index] = data[lab[i] + combin[i]][index] + residential * w

    ######## MCL #############################################################################################
    # compute corrected values: MCL_####_P/E/R/S
    # lab = ['P', 'E', 'R', 'S']
    # met = ['PR', 'CKF', 'MCL', 'MSD']
    for combin in Combinations: # combination numbers #### 2114
        # get observations
        d = []       # distances of the Combinations ####: [PES]_d#t
        params = []  # values of the combination P,E,R,S
        for i in [0, 1, 2, 3]:
            dis = MCL_distances[lab[i] + combin[i]]
            d.append(dis)

            params.append(data[lab[i] + combin[i]][index])

        residential = params[0] - params[1] - params[2] - params[3]
        # compute weights w and MCL corrected values        
        for i in [0, 1, 2, 3]:
            w = d[i] / sum(d)

            data['MCL_' + combin + '_' + lab[i] + '_w'][index] = w
            if i == 0:
                w = -w
            data['MCL_' + combin + '_' + lab[i]][index] = data[lab[i] + combin[i]][index] + residential * w

    ######## MSD #############################################################################################
    # distances of all observations
    Ds = []
    # compute D: P/E/R/S#_D
    for rsName in colFiltered:
        x = data[rsName]  # P1 remote sensing observations
        y = data[rsName[0]]  # P  true values

        if not first:
            y_true = y[0:(index + 1)]
            y_true_mean = y_true.mean()
            y_rs = x[0:(index + 1)]
            y_rs_mean = y_rs.mean()

            # compute distance
            data[rsName + '_D'][index] = -(((y_true - y_true_mean) * (y_rs - y_rs_mean)).sum()) ** 2 / (
                        (y_rs - y_rs_mean) ** 2).sum() + ((y_true - y_true_mean) ** 2).sum()

        Ds.append(data[rsName + '_D'][index])
    
    # compute w and corrected values: MSD_P/E/R/S_w, MSD_####_P/E/R/S
    for combin in Combinations:
        # get distances
        d = []       # distances of the combination
        params = []  # values of the combination P,E,R,S
        for i in [0, 1, 2, 3]:
            d.append(data[lab[i] + combin[i] + '_D'][index])
            params.append(data[lab[i] + combin[i]][index])

        residential = params[0] - params[1] - params[2] - params[3]
        # compute weights w and MSD corrected values
        for i in [0, 1, 2, 3]:
            if first:
                w = abs(params[i]) / sum([abs(ele) for ele in params])
            else:
                w = d[i] / sum(d)

            data['MSD_' + combin + '_' + lab[i] + '_w'][index] = w
            if i == 0:
                w = -w
            data['MSD_' + combin + '_' + lab[i]][index] = data[lab[i] + combin[i]][index] + residential * w

    first = False
# print('Computed BCC adjusted water budget:',data)
print('################ Saved BCC to csv #######################################')
if introObs:
    data.to_csv(outputDir+'data_introObs_bcc.csv')
else:
    data.to_csv(outputDir+'data_bcc.csv')

# ##### validate with true values ################################################################
# # Compute pbias, cc, rse, me, mae, mape for remotesensing PERS, method_P/E/R/S_####_bias/cc/rse
# individualStat = pd.Series()
# # Traverse stats
# for stat in statistics:
#     # remote sensing data
#     # Compute pbias, cc, rse, me, mae, mape for remote sensing data
#     # Traverse P/E/R/S
#     for m in lab:
#         # print("In ", m, stat)
#         r = re.compile(m+"\d")
#         rsFiltered = list(filter(r.match, colFiltered))  # ['P1', 'E1', 'R1'...]
#         # print("filtered columns",rsFiltered)

#         for o in rsFiltered:
#             # abs
#             t = data[m+"_closed"][3:]
#             c = data[o][3:]

#             s = globals()['compute'+stat](t,c)
#             individualStat[o+'_'+stat] = s      # P1_PBIAS
#             # print("In ", o, s)

#     # Compute pbias, cc, rse, me, mae, mape for combinations
#     # Traverse method:
#     for method in met:
#         # Traverse P/E/R/S
#         for m in lab:
#             # Traverse combinations: 1111, 1112,...
#             for combin in Combinations:
#                 trueValues = data[m+'_closed'][3:]
#                 correctedValues = data[method+'_'+combin+'_'+m][3:]

#                 s = globals()['compute'+stat](trueValues, correctedValues)
#                 individualStat[method+'_'+m+'_'+combin+'_'+stat] = s      # PR_P_1111_PBIAS

# # print("individualStat",individualStat)
# df_stat = individualStat
# df_stat['bestComb'] = '2131' #combBest
# # print("df_stat",df_stat)
# # df_stat.to_csv(outputDir+'estimate_stats.csv')
# print('#######################################################')

# # comparison
# df_stat_ind = df_stat
# statCols = df_stat.keys().values
# tab_best = pd.DataFrame()
# tab_all = pd.DataFrame()
# computeDict={
#     'PBIAS':computePBIAS,
#     'CC':computeCC,
#     'RMSE':computeRMSE,
#     'ME':computeME,
#     'ME1':computeME1,
#     'MAE':computeMAE,
#     'MAPE':computeMAPE
# }
# # Out stats
# # P_PBIAS, P_PR_PBIAS
# # Traverse P/E/R/S
# for i in [0,1,2,3]:
#     # remote sensing stats
#     # best
#     t0 = pd.Series()
#     t0['index'] = lab[i]+'_'+lab[i]
#     # bestRaw
#     t_bestRaw = pd.Series()
#     t_bestRaw['index'] = lab[i]+'_bestRaw'
#     # EO
#     t_EO = pd.Series()
#     t_EO['index'] = lab[i]+'_EO'

#     # all
#     t1 = pd.Series()
#     t1['index'] = lab[i]+'_'+lab[i]

#     # Traverse stats
#     for stat in statistics:
#         #############################
#         # Best
#         mat = df_stat_ind[lab[i]+str(df_stat_ind['bestComb'])[i]+"_"+stat]
#         t0[stat] = mat#.mean()
#         # bestRaw
#         t_bestRaw[stat] = computeDict[stat](data[lab[i]+'_closed'],data[lab[i]+combBest[i]])        
#         # EO
#         t_EO[stat] = computeDict[stat](data[lab[i]+'_closed'],data[lab[i]])        

#         #############################
#         # All
#         # filter columns
#         r = re.compile(lab[i] + "[1-5]{1}(?!\d)" + "_" + stat+"$")  # 'P'+"[12345](?!\d)$"
#         statFiltered = list(filter(r.match, statCols))
#         t1[stat] = df_stat_ind[statFiltered].mean()
 
#     tab_best = pd.concat([tab_best,t0,t_bestRaw,t_EO], ignore_index=True)
#     tab_all = pd.concat([tab_all,t1], ignore_index=True) 
#     # print(tab, t0)
# # print(tab)   

#     # Corrected value stats
#     # Traverse method:
#     for method in met:
#         # best
#         t2 = pd.Series()
#         # all
#         t3 = pd.Series()
#         # Traverse stats
#         for stat in statistics:
#             # print("In ",m,method,stat)
#             t2['index'] = lab[i] + '_' + method
#             t3['index'] = lab[i] + '_' + method

#             #############################
#             # Best                 
#             mat = df_stat_ind[method+'_'+lab[i]+'_'+str(df_stat_ind['bestComb'])+"_"+stat]
#             t2[stat] = mat

#             #############################
#             # All
#             # filter columns
#             # match combinations
#             r = re.compile(method+'_'+lab[i]+'_'+"[1-5]{4}(?!\d)"+"_"+stat+"$") #PR_P_1111_PBIAS
#             statFiltered = list(filter(r.match, statCols))
#             t3[stat] = df_stat_ind[statFiltered].mean()
#             # print('mean', t[stat])
                        
#         tab_best = pd.concat([tab_best,t2], ignore_index=True)  #tab_best.append(t2, ignore_index=True)
#         tab_all = pd.concat([tab_all,t3], ignore_index=True)  #tab_all.append(t3, ignore_index=True)

# print('################################### tab_best\n',tab_best)
# print('################################### tab_all\n',tab_all)
# tab_best.to_csv(outputDir+'estimate_BestCombination_BestEO.csv')
# tab_best.to_csv(outputDir+'estimate_BestCombination_2131.csv')
# tab_best.to_csv(outputDir+'estimate_BestCombination.csv')
# tab_all.to_csv(outputDir+'estimate_AllCombination.csv')

# # visualization
# target = 'RMSE'
# tab = tab_best
# r = pd.DataFrame()    
# # raw
# raw = pd.Series()
# raw['BCCs'] = 'Raw'
# for c in lab:
#     raw['Components'] = c
#     raw[target] = tab[tab['index']==c+'_'+c][target].values[0]
#     r = r.append(raw, ignore_index=True)
# # BCCs
# for c in lab: # ['P','E','R','S']
#     for m in met:   # ['PR','KCF','MCL','MSD']
#         ind = pd.Series()
#         ind['Components'] = c
#         ind['BCCs'] = m
#         # print(c,m,'RMSE values',tab[tab['index']==c+'_'+m][target].values[0])
#         ind[target] = tab[tab['index']==c+'_'+m][target].values[0]
#         # print('ind',ind)

#         r = r.append(ind, ignore_index=True)
#         # print(r)
# # print(r)
# r.Components=pd.Categorical(r.Components,categories=r.Components.unique(),ordered=True)
# r.BCCs=pd.Categorical(r.BCCs,categories=r.BCCs.unique(),ordered=True)
# r.pivot(index='Components', columns='BCCs', values=target).plot(kind='bar')

# plt.ylabel(target)
# plt.show()