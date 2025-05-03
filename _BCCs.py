# from curses import A_CHARTEXT
import os, fnmatch
import pandas as pd
import re
import numpy as np
from itertools import combinations
from globVar import basin3Flag, find_pattern, get_file_name,getMergedTrue,generateAInverse,getSquaredDistance,getUncertCoefR

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None

test = True
# # # 3 test basins 
# pth = os.path.join(os.path.dirname(__file__), '', 'dataTWSC_test/')
# output_dir = os.path.join(os.path.dirname(__file__), '', 'output_test/')
# # # fl = pth +'3629001.csv'
csv_folder = os.path.join(os.path.dirname(__file__), '', '')
if basin3Flag:    
    # # 3 test basins 
    pth = os.path.join(os.path.dirname(__file__), '', '3data_basin/')
    output_dir = os.path.join(os.path.dirname(__file__), '', '3BasinsComparison/')
else:
    pth = os.path.join(os.path.dirname(__file__), '', '28data_basin/')
    output_dir = os.path.join(os.path.dirname(__file__), '', '28BasinsComparison/')

if test:
    # pattern = '6742900.csv'
    # pattern = '1147010.csv'
    # pattern = '1147010_test.csv'
    pattern = '4127800.csv'
else:
    pattern = "*.csv"
fileList = find_pattern(pattern, pth)
# print(fileList)

lab = ['P', 'E', 'R', 'S']
met = ['PR', 'CKF', 'MCL', 'MSD']
for fl in fileList:
    fileName = get_file_name(fl)
    print("/////////////////////////////////////////////////////////////////",fileName)

    # df is existing data for validating if BCC functions are right
    # data is used for computing BCC corrected water budget components
    if test:
        data = pd.read_csv(fl)#.head(10)
        # print('data\n',data)
    else:
        data = pd.read_csv(fl) 

    if not basin3Flag:
        columns_to_convert = ['GRACE_CSR', 'GRACE_GFZ','GRACE_JPL']
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    if basin3Flag:        
        # rename
        data = data.drop(['SM_GLDAS','SWE_GLDAS'], axis=1)
        data.rename(columns={'Pre_GPCC':'P1','Pre_GPCP':'P2','Pre_Gsmap':'P3','Pre_IMERG':'P4','Pre_PERSIANN_CDR':'P5',
                        'ET_FLUXCOM':'E1','ET_GLDAS':'E2','ET_GLEAM':'E3','ET_PT-JPL':'E4',
                        'TWSC_GRACE_CSR_calculate':'S1','TWSC_GRACE_GFZ_calculate':'S2','TWSC_GRACE_JPL_calculate':'S3','TWSC_GRACE_Mascon_JPL_calculate':'S4',
                        'GRDC':'R1','Unnamed: 0':'date'},inplace=True)
    else:
        data.rename(columns={'P_GPCC':'P1','P_GPM':'P2','P_MSWEP':'P3','P_PERSIANN':'P4',
                        'ET_ERA5':'E1','ET_GLEAM':'E2','ET_MERRA':'E3',
                        'GRACE_CSR':'S1','GRACE_GFZ':'S2','GRACE_JPL':'S3',
                        'GRDC':'R1','date':'date'},inplace=True)

    # ####################################
    # # add columns to compute error limit  
    # # according to true values E_P/E/R/S#
    # # ##################################
    # # For P: true value Pre_GPCC
    # data['E_P1'] = abs(data['P1']-data['Pre_GPCC'])
    # data['E_P2'] = abs(data['P2']-data['Pre_GPCC'])
    # data['E_P3'] = abs(data['P3']-data['Pre_GPCC'])
    # # For E: E#*20%
    # data['E_E1'] = data['E1']*0.2
    # data['E_E2'] = data['E2']*0.2
    # data['E_E3'] = data['E3']*0.2
    # # For R: R*7%
    # data['E_R1'] = data['R1']*0.07
    # # For S: true value TWSC_GRACE_Mascon_JPL_calculate
    # data['E_S1'] = abs(data['S1']-data['TWSC_GRACE_Mascon_JPL_calculate'])
    # data['E_S2'] = abs(data['S2']-data['TWSC_GRACE_Mascon_JPL_calculate'])
    # data['E_S3'] = abs(data['S3']-data['TWSC_GRACE_Mascon_JPL_calculate'])
    # # print('error limit',data)

    ##############################################################################################################
    # Preprocessing to get metadata
    # columns and combination
    col = data.columns
    # match remote sensing observations
    r = re.compile("[PERS]\d")
    colFiltered = list(filter(r.match, col))  # ['P1', 'E1', 'R1'...]        

    # match combinations
    exhaustCompnents = [] # for P,E,S [['1', '2', '3', '4', '5'], ['1', '2', '3', '4'], ['1', '2', '3', '4']]
    for m in ['P','E','S']: 
        r = re.compile(m+"\d")
        _colFiltered = list(filter(r.match, col))
        # print(colFiltered)
        exhaustCompnents.append([s[1:] for s in _colFiltered])  # For P: ['1', '2', '3', '4', '5']
    # print('exhaustCompnents',exhaustCompnents)
    Combinations = [a+b+'1'+c for a in exhaustCompnents[0] for b in exhaustCompnents[1] for c in exhaustCompnents[2]]  # ['1111', '1112'...]
    # print('Combinations',Combinations,len(Combinations))

    # give data true columns: P/E/R/S
    for m in lab:
        data[m] = -9999.0    

    # give data D columns: P/E/R/S#_D
    # give data error limit columns: E_P/E/R/S#
    for colF in colFiltered:
        data[colF + '_D'] = -9999.0
        data['E_'+colF] = -9999.0
    # print('data\n',data.columns)

    # give data w columns: PR/CKF/MCL/MSD_####_P/E/R/S_w
    # give data corrected value columns: MSD_####_P/E/R/S
    for combin in Combinations:
        for m in lab:
            for k in met:
                data[k + '_' + combin + '_' + m] = -9999.0
                data[k + '_' + combin + '_' + m + '_w'] = -9999.0
                data[k + '_' + combin + '_' + m + '_r'] = -9999.0
    # print('data\n',data.columns)

    ################################################################################################################
    # Preprocessing to get true values P,E,S and
    # matrices definition for computing the parameter distances of MCL: 
    # MCL_distances 12 values (5 3 4 for P ET and S respectively)
    # compute true values
    for index, row in data.iterrows():
        for m in ['P', 'E', 'S']:
            # raw
            r = re.compile(m + "[12345](?!\d)$")  # code : 12345
            filtered = list(filter(r.match, col))

            arr = data[filtered].iloc[index].to_numpy()
            data[m][index] = getMergedTrue(arr)
            ####################################
            # add columns to compute error limit  
            # according to true values E_P/E/R/S#
            # ##################################
            for column in filtered:
                data['E_'+column][index] = abs(data[column][index]-data[m][index])

        data['R'][index] = data['R1'][index]
        # Error limit for R
        data['E_R1'][index] = data['R1'][index]*0.07
    # print('with true values', data[['P', 'E', 'S']])

    # Construct dictionary for  [PES]_d#t 
    # e.g. P has 5 distances, one for each product, i.e. P_d[1-5]t
    # distances of all observations
    MCL_distances = {}  # 'P2':3
    for m in ['P', 'E', 'S']:#
        r = re.compile(m+"\d")
        param = list(filter(r.match, colFiltered))
        paramNum = len(param)
        # print('-------------------------param',param, paramNum)

        # A inverse (Eq.9 in Pan, RSE15)
        A_1 = generateAInverse(paramNum)
        # iterate combinations for columns (get y with eq. 10 in Pan, RSE15)
        djk = []
        combin = combinations(np.arange(1,paramNum+1),2)
        for c in combin:            
            j = c[0]-1
            k = c[1]-1       

            tempData = data[[param[j],param[k]]].dropna()
            djk += [getSquaredDistance(tempData[param[j]],tempData[param[k]])]
        
        djk = np.array(djk)[np.newaxis]
        y = np.dot(np.array(A_1,dtype=object), djk.T)

        # get root and convert lists to dictionary
        res = {param[i]: np.power(y[i][0],0.5) for i in range(len(param))}

        MCL_distances = MCL_distances|res
    print('MCL_distances\n',MCL_distances)

    ################################################################################################################
    # iterate rows to compute D, w, and corrected values
    first = True
    for index, row in data.iterrows():
        # print("------------In row: "+str(index)+"------------------------------------------------")
        ######## PR #############################################################################################
        # compute corrected values: PR_####_P/E/R/S
        # lab = ['P', 'E', 'R', 'S']
        # met = ['PR', 'CKF', 'MCL', 'MSD']
        # print("------------In PR------------------------------------------------")
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
                data['PR_' + combin + '_' + lab[i]+'_r'][index] = residential * w

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
                # replace runoff distance with R1_D
                if i == 2:
                    param = getUncertCoefR(data['R1'],index)
                    dis = np.power(data['R1'][index] * param,2)
                d.append(dis)
                params.append(data[lab[i] + combin[i]][index])
            # runoff should have the lowest weight
            d[2] = np.min(d)

            residential = params[0] - params[1] - params[2] - params[3]
            # compute weights w and CKF corrected values        
            for i in [0, 1, 2, 3]:
                w = d[i] / sum(d)

                data['CKF_' + combin + '_' + lab[i] + '_w'][index] = w
                if i == 0:
                    w = -w
                data['CKF_' + combin + '_' + lab[i]][index] = data[lab[i] + combin[i]][index] + residential * w
                data['CKF_' + combin + '_' + lab[i]+'_r'][index] = residential * w

        ######## MCL #############################################################################################
        # compute corrected values: MCL_####_P/E/R/S
        # lab = ['P', 'E', 'R', 'S']
        # met = ['PR', 'CKF', 'MCL', 'MSD']
        for combin in Combinations: # combination numbers #### 2114
            # get observations
            d = []       # distances of the Combinations ####: [PES]_d#t
            params = []  # values of the combination P,E,R,S
            for i in [0, 1, 2, 3]:
                if i != 2:
                    dis = MCL_distances[lab[i] + combin[i]]
                else:
                    param = getUncertCoefR(data['R1'],index)
                    dis = data['R1'][index] * param
                d.append(dis)

                params.append(data[lab[i] + combin[i]][index])
            # runoff should have the lowest weight
            d[2] = np.min(d)

            residential = params[0] - params[1] - params[2] - params[3]
            # compute weights w and MCL corrected values        
            for i in [0, 1, 2, 3]:
                w = d[i] / sum(d)

                data['MCL_' + combin + '_' + lab[i] + '_w'][index] = w
                if i == 0:
                    w = -w
                data['MCL_' + combin + '_' + lab[i]][index] = data[lab[i] + combin[i]][index] + residential * w
                data['MCL_' + combin + '_' + lab[i]+'_r'][index] = residential * w

        ######## MSD #############################################################################################
        # distances of all observations
        Ds = []
        # compute D: P/E/R/S#_D
        for rsName in colFiltered:
            x = data[rsName]  # P1 remote sensing observations
            y = data[rsName[0]]  # P  true values

            if rsName[0] != 'R':
                # P E S
                if not first:
                    y_true = y[0:(index + 1)]
                    y_true_mean = y_true.mean()
                    y_rs = x[0:(index + 1)]
                    y_rs_mean = y_rs.mean()

                    # compute distance
                    data[rsName + '_D'][index] = -(((y_true - y_true_mean) * (y_rs - y_rs_mean)).sum()) ** 2 / (
                                (y_rs - y_rs_mean) ** 2).sum() + ((y_true - y_true_mean) ** 2).sum()
            else: # runoff
                param = getUncertCoefR(x,index)
                data['R1_D'][index] = y[index] * param

            Ds.append(data[rsName + '_D'][index])

        # The uncertainty of R is always the lowest
        data['R1_D'][index] = min(Ds)
        
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
                data['MSD_' + combin + '_' + lab[i]+'_r'][index] = residential * w

        first = False

        # # test 
        # testParam = '3111_P'
        # print('True\n',df[['PR_'+testParam,'CKF_'+testParam,'MCL_'+testParam,'MSD_'+testParam]])
        # print('Computed\n',data[['PR_'+testParam,'CKF_'+testParam,'MCL_'+testParam,'MSD_'+testParam]])

    # # test
    # if test:    
    #     print("-----examine results-----------------------------------")
    #     r = re.compile(".*_[PER]$")
    #     colFiltered = list(filter(r.match, data.columns))  # ['1111_P_w']
    #     print(colFiltered)
    #     tdf = data[colFiltered]
    #     print(tdf)
    #     t = tdf.drop([tdf.index[0], tdf.index[1], tdf.index[2]])
    #     print(t.min())
    #     print(t.min().loc[lambda x : x < 0])

    ######## drop columns and save files ############################################
    # # drop w columns: ####_P/E/R/S_w
    # r = re.compile(".*_w")
    # colFiltered = list(filter(r.match, data.columns))  # ['1111_P_w']
    # data = data.drop(colFiltered, axis=1)

    # # drop D (P1_D)
    # r = re.compile(".*_D")
    # colFiltered = list(filter(r.match, data.columns))
    # data = data.drop(colFiltered, axis=1)

    # save dataTest
    if test:
        data.to_csv(output_dir+get_file_name(fl)+'_bcc.csv',index=False)
    else:
        data.to_csv(output_dir+get_file_name(fl)+'_bcc'+fl[-4:],index=False)
