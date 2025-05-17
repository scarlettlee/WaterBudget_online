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
from globVar import compute_stats, getMergedTrue, getSquaredDistance, generateAInverse # Added more from globVar
import seaborn as sns

mpl.rcParams['axes.linewidth'] = 1.5 #set the value globally
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
simplefilter(action="ignore", category=FutureWarning)

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

# --- Scenario Control Parameters ---
OVER_ADJUST_PERC = 0.5 # Example: 50% over adjustment allowed beyond uncertainty
TOLERANCE_OBS_INTRO = 0.15 # Example: 15% deviation for P before forcing to P_closed

# get component num order
def getNumOrder(m):
    r = re.compile(m + "[12345](?!\d)$")
    filtered = list(filter(r.match, col))

    e = [i[1] for i in filtered]
    return e

def sign(x):
  if x > 0: return 1
  elif x < 0: return -1
  else: return 0

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
month_count = 60 # Number of time steps
t_time = np.linspace(0, 20*np.pi, month_count) # Changed variable name from t to t_time
# P = am*sin(w*t+bp) + 3
# ET = A*sin(w*t+a)  + 3
# R = B*sin(w*t+b)  + 1
# S = C*sin(w*t+c)  - 1
P = A*sin(w*t_time+a)  + 3.5
ET = B*sin(w*t_time+b)  + 3
R = C*sin(w*t_time+c)  + 1.5
S = am*sin(w*t_time+bp) - 1

###### generate observations ##################################################################
np.random.seed(0)
# 0 is the mean of the normal distribution you are choosing from
# 1 is the standard deviation of the normal distribution
# month_count is the number of elements you get in array noise
noise = np.random.normal(0,0.1,month_count)
P1 = P+noise+1 # Added some bias
# np.random.seed(2)
noise = np.random.normal(0,0.2,month_count)
P3 = P+noise+1.5
noise = np.random.normal(0,0.3,month_count)
P4 = P+noise+2
noise = np.random.normal(0,0.4,month_count)
P5 = P+noise+0.5
np.random.seed(6)
noise = np.random.normal(0,0.1,month_count)
P2 = P+noise+1.2
# Artificial system error
P2[58] = 20

np.random.seed(1)
noise = np.random.normal(0,0.1,month_count)
ET1 = ET+noise+0.5
noise = np.random.normal(0,0.2,month_count)
ET2 = ET+noise+1
noise = np.random.normal(0,0.3,month_count)
ET3 = ET+noise-0.5

np.random.seed(2)
noise = np.random.normal(0,0.05,month_count) # Runoff usually has smaller relative error
R1 = R+noise+0.1
noise = np.random.normal(0,0.03,month_count)
R2 = R+noise-0.1
noise = np.random.normal(0,0.02,month_count)
R3 = R+noise

np.random.seed(3)
noise = np.random.normal(0,0.2,month_count)
S1 = S+noise+0.2
noise = np.random.normal(0,0.3,month_count)
S2 = S+noise-0.2
noise = np.random.normal(0,0.4,month_count)
S3 = S+noise

noise1 = np.random.normal(0,0.1,month_count)
noise2 = np.random.normal(0,0.2,month_count)
noise3 = np.random.normal(0,0.3,month_count)
noise4 = np.random.normal(0,0.4,month_count)

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

# plt.figure(figsize=(12,8))
# plt.plot(t,P1,'--',label ='P2', alpha=.6)
# plt.plot(t,P1,'--',label ='P3', alpha=.6)
# plt.plot(t,P1,'--',label ='P4', alpha=.6)
# plt.plot(t,P1,'--',label ='P5', alpha=.6)

# plt.plot(t_time,P,label ='P (True)')
# plt.plot(t_time,ET,label ='ET (True)')
# plt.plot(t_time,R,label ='R (True)')
# plt.plot(t_time,S,label ='TWSC (True)')

# plt.plot(t_time,noise1,'--',label ='ns 0.1')
# plt.plot(t_time,noise2,'--',label ='ns 0.2')
# plt.plot(t_time,noise3,'--',label ='ns 0.3')
# plt.plot(t_time,noise4,'--',label ='ns 0.4')

# plt.plot(t_time,P-ET-R-S,label = r'$\Delta Res_{true}$', linewidth=2) # Should be zero

# plt.xlabel('Month')
# plt.ylabel('Value')
# plt.title('Synthetic True Water Budget Components and Noise')
# plt.legend(framealpha=1, loc='upper right')
# plt.show()

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
'P_merged':np.average([P1,P2,P3,P4,P5],axis=0), # merged P
'E1':ET1,
'E2':ET2,
'E3':ET3,
'E_closed':ET,
"E_merged":np.average([ET1,ET2,ET3],axis=0), # merged E
'R1':R1,
'R2':R2,
'R3':R3,
'R_closed':R,
"R_merged":np.average([R1,R2,R3],axis=0), # merged R
'S1':S1,
'S2':S2,
'S3':S3,
'S_closed':S,
"S_merged":np.average([S1,S2,S3],axis=0), # merged S
}
data = pd.DataFrame(dict)
data_initial = data.copy() # Keep a pristine copy

if test:
    data_initial = data_initial.head(6)
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
os.makedirs(outputDir, exist_ok=True) # Ensure the output directory exists
# data_initial.to_csv(outputDir+'oriData_synthetic.csv') # Save initial synthetic products
# generate combinations
Combinations = [] # ['1111', '1112'...]
for e1,e2,e3,e4 in itertools.product(getNumOrder('P'), getNumOrder('E'), getNumOrder('R'), getNumOrder('S')):
    Combinations += [e1+e2+e3+e4]   
# print('combinations\n',Combinations)

summary_stats_filepath = os.path.join(outputDir, 'synthetic_aggregated_stats_summary.csv')
all_results_filepath = os.path.join(outputDir, 'synthetic_all_scenarios_results.csv')

if os.path.exists(summary_stats_filepath) and os.path.exists(all_results_filepath):
    print(f"Loading results from cache: {summary_stats_filepath} and {all_results_filepath}")
    all_scenarios_stats_summary = pd.read_csv(summary_stats_filepath)
    processed_data_all_scenarios = pd.read_csv(all_results_filepath)
else:      
    # match remote sensing observations
    r = re.compile("[PERS]\d")
    colFiltered = list(filter(r.match, col))  # ['P1', 'E1', 'R1'...]

    # --- Pre-calculate MCL distances (once for all scenarios) ---
    MCL_distances = {}  # 'P2':3
    for m in lab:
        r = re.compile(m+"\d")
        param = list(filter(r.match, colFiltered))
        paramNum = len(param)

        A_1 = generateAInverse(paramNum)
        djk = []
        combin_mcl = combinations(np.arange(1,paramNum+1),2)
        for c in combin_mcl:
            j = c[0]-1
            k = c[1]-1       
            # Use data_initial for MCL distances, as they are based on inter-product differences
            djk += [getSquaredDistance(data_initial[param[j]],data_initial[param[k]])] 
        
        djk = np.array(djk)[np.newaxis]
        y = np.dot(np.array(A_1,dtype=object), djk.T)
        res = {param[i]: np.power(abs(y[i][0]),0.5) for i in range(len(param))}
        MCL_distances = MCL_distances|res
    print('MCL_distances\n',MCL_distances)


    scenarios = [
        {"name": "OrigBCC", "intro_obs": False, "solve_outliers": False, "solve_overadjust": False},
        {"name": "ObsIntro", "intro_obs": True, "solve_outliers": False, "solve_overadjust": False},
        {"name": "ObsOutlier", "intro_obs": True, "solve_outliers": True, "solve_overadjust": False},
        {"name": "ObsOutlierOver", "intro_obs": True, "solve_outliers": True, "solve_overadjust": True},
    ]

    processed_data_all_scenarios = data_initial.copy()

    # --- Pre-calculate "merged" E, R, S for CKF/MSD reference if not using true P ---
    for m_comp in ['E', 'S']: # P and R are handled differently or are true
        r_m = re.compile(m_comp + "[12345](?!\d)$")
        m_filtered_cols = list(filter(r_m.match, processed_data_all_scenarios.columns))
        processed_data_all_scenarios[m_comp + '_merged'] = processed_data_all_scenarios[m_filtered_cols].mean(axis=1) # Simple mean for merged

    # --- Pre-calculate Product Uncertainties (E_P1, E_E1, etc.) ---
    for comp_label_idx, comp_label_val in enumerate(lab): # P, E, R, S
        r_prod = re.compile(comp_label_val + r"[12345](?!\d)$")
        prod_cols_for_comp = list(filter(r_prod.match, processed_data_all_scenarios.columns))
        for prod_col in prod_cols_for_comp:
            processed_data_all_scenarios['E_' + prod_col] = abs(processed_data_all_scenarios[prod_col] - processed_data_all_scenarios[comp_label_val + '_closed'])


    for scenario_config in scenarios:
        scenario_name = scenario_config["name"]
        print(f"Processing Scenario: {scenario_name}")
        
        # Create columns for this scenario's results
        for combin_scen in Combinations:
            for comp_scen in lab:
                for meth_scen in met:
                    processed_data_all_scenarios[f'{meth_scen}_{combin_scen}_{comp_scen}_{scenario_name}'] = np.nan
                    processed_data_all_scenarios[f'{meth_scen}_{combin_scen}_{comp_scen}_w_{scenario_name}'] = np.nan # Weights might change
                    processed_data_all_scenarios[f'{meth_scen}_{combin_scen}_{comp_scen}_r_{scenario_name}'] = np.nan # Adjustments

        # --- Iterate rows to compute D, w, and corrected values ---
        first_msd = True # For MSD method's first step
        for index, row_data in processed_data_all_scenarios.iterrows():
            # --- MSD's D value calculation (cumulative) ---
            if not first_msd:
                for rsName_msd in colFiltered: # P1, P2... E1, E2...
                    y_true_series = processed_data_all_scenarios[rsName_msd[0]+'_closed'].iloc[0:(index + 1)]
                    y_rs_series = processed_data_all_scenarios[rsName_msd].iloc[0:(index + 1)]
                    
                    y_true_mean_msd = y_true_series.mean()
                    y_rs_mean_msd = y_rs_series.mean()

                    numerator_msd = -(((y_true_series - y_true_mean_msd) * (y_rs_series - y_rs_mean_msd)).sum()) ** 2
                    denominator_msd = ((y_rs_series - y_rs_mean_msd) ** 2).sum()
                    term1_msd = numerator_msd / denominator_msd if denominator_msd != 0 else 0
                    term2_msd = ((y_true_series - y_true_mean_msd) ** 2).sum()
                    processed_data_all_scenarios.loc[index, rsName_msd + '_D'] = term1_msd + term2_msd
            else: # first row for MSD
                for rsName_msd in colFiltered:
                    processed_data_all_scenarios.loc[index, rsName_msd + '_D'] = 0 # Or some initial large value

            for method_name in met:
                for comb_str in Combinations:
                    # 1. Initial BCC Application
                    params_val = [row_data[lab[i] + comb_str[i]] for i in range(4)] # P,E,R,S from selected products
                    
                    # Determine reference P for CKF/MSD based on scenario
                    p_ref_ckf_msd = row_data['P_closed'] if scenario_config["intro_obs"] else row_data['P_merged'] # P_merged needs to be precalculated
                    e_ref_ckf_msd = row_data['E_merged']
                    s_ref_ckf_msd = row_data['S_merged']
                    r_ref_val = row_data['R1'] # R is usually taken as is or from a single product

                    weights = [0.0] * 4
                    if method_name == 'PR':
                        sum_abs_params = sum(abs(p) for p in params_val if pd.notna(p))
                        if sum_abs_params != 0:
                            weights = [abs(params_val[i]) / sum_abs_params if pd.notna(params_val[i]) else 0 for i in range(4)]
                    elif method_name == 'CKF':
                        d_ckf = [
                            (params_val[0] - p_ref_ckf_msd)**2 if pd.notna(params_val[0]) and pd.notna(p_ref_ckf_msd) else np.inf,
                            (params_val[1] - e_ref_ckf_msd)**2 if pd.notna(params_val[1]) and pd.notna(e_ref_ckf_msd) else np.inf,
                            (params_val[2] - r_ref_val)**2 if pd.notna(params_val[2]) and pd.notna(r_ref_val) else np.inf, # Or a fixed uncertainty for R
                            (params_val[3] - s_ref_ckf_msd)**2 if pd.notna(params_val[3]) and pd.notna(s_ref_ckf_msd) else np.inf
                        ]
                        sum_d_ckf = sum(d for d in d_ckf if d != np.inf)
                        if sum_d_ckf != 0:
                            weights = [d / sum_d_ckf if d != np.inf else 0 for d in d_ckf]
                    elif method_name == 'MCL':
                        d_mcl = [MCL_distances.get(lab[i] + comb_str[i], np.inf) for i in range(4)]
                        sum_d_mcl = sum(d for d in d_mcl if d != np.inf)
                        if sum_d_mcl != 0:
                            weights = [d / sum_d_mcl if d != np.inf else 0 for d in d_mcl]
                    elif method_name == 'MSD':
                        if first_msd: # Use PR weights for the first step of MSD
                            sum_abs_params = sum(abs(p) for p in params_val if pd.notna(p))
                            if sum_abs_params != 0:
                                weights = [abs(params_val[i]) / sum_abs_params if pd.notna(params_val[i]) else 0 for i in range(4)]
                        else:
                            d_msd = [row_data.get(lab[i] + comb_str[i] + '_D', np.inf) for i in range(4)]
                            sum_d_msd = sum(d for d in d_msd if d != np.inf and pd.notna(d))
                            if sum_d_msd != 0:
                                weights = [d / sum_d_msd if d != np.inf and pd.notna(d) else 0 for d in d_msd]

                    # Store weights
                    for i in range(4):
                        processed_data_all_scenarios.loc[index, f'{method_name}_{comb_str}_{lab[i]}_w_{scenario_name}'] = weights[i]

                    # Calculate residual and initial corrected values
                    p_for_residual = row_data['P_closed'] if scenario_config["intro_obs"] else params_val[0]
                    if any(pd.isna(val) for val in [p_for_residual, params_val[1], params_val[2], params_val[3]]):
                        residual = np.nan
                    else:
                        residual = p_for_residual - params_val[1] - params_val[2] - params_val[3]

                    current_corrected = [np.nan]*4
                    current_adjustments = [np.nan]*4

                    if pd.notna(residual):
                        for i in range(4):
                            adj_amount = residual * weights[i]
                            current_adjustments[i] = -adj_amount if i == 0 else adj_amount # P adjustment is -(residue*weight)
                            current_corrected[i] = params_val[i] + current_adjustments[i] if pd.notna(params_val[i]) else np.nan
                    
                    # --- Scenario 2: Introduce Observations (P) ---
                    if scenario_config["intro_obs"]:
                        p_target_obs = row_data['P_closed']
                        if pd.notna(p_target_obs) and pd.notna(current_corrected[0]) and p_target_obs != 0:
                            if abs(current_corrected[0] - p_target_obs) / abs(p_target_obs) > TOLERANCE_OBS_INTRO:
                                iter_res_p_obs = current_corrected[0] - p_target_obs
                                current_corrected[0] = p_target_obs
                                current_adjustments[0] -= iter_res_p_obs # Adjustment for P changed
                                
                                # Redistribute iter_res_p_obs to E, R, S
                                weights_ers = [weights[1], weights[2], weights[3]]
                                sum_weights_ers = sum(w for w in weights_ers if pd.notna(w))
                                if sum_weights_ers != 0:
                                    for i_ers in range(1, 4): # E, R, S
                                        if pd.notna(weights[i_ers]) and pd.notna(current_corrected[i_ers]):
                                            adj_share = iter_res_p_obs * (weights[i_ers] / sum_weights_ers)
                                            current_corrected[i_ers] -= adj_share 
                                            current_adjustments[i_ers] -= adj_share

                    # --- Scenario 3: Solve Outliers (Sign Flips for P, E, S) ---
                    if scenario_config["solve_outliers"]:
                        total_outlier_redist_val = 0
                        components_for_outlier_redist = {0,1,2,3} # Indices for P,E,R,S
                        for i_outlier in [0, 1, 3]: # P, E, S
                            comp_true_val = row_data[lab[i_outlier]+'_closed']
                            if pd.notna(current_corrected[i_outlier]) and pd.notna(comp_true_val) and comp_true_val != 0:
                                if sign(current_corrected[i_outlier]) != sign(comp_true_val):
                                    delta_outlier = current_corrected[i_outlier] - comp_true_val # Amount to remove from budget
                                    current_corrected[i_outlier] = comp_true_val
                                    current_adjustments[i_outlier] -= delta_outlier
                                    total_outlier_redist_val += delta_outlier
                                    if i_outlier in components_for_outlier_redist:
                                        components_for_outlier_redist.remove(i_outlier)
                        
                        # Redistribute total_outlier_redist_val
                        active_weights_outlier = [weights[i] for i in components_for_outlier_redist if pd.notna(weights[i])]
                        sum_active_weights_outlier = sum(active_weights_outlier)
                        if sum_active_weights_outlier != 0 and total_outlier_redist_val !=0:
                            for i_redist_outlier in components_for_outlier_redist:
                                if pd.notna(weights[i_redist_outlier]) and pd.notna(current_corrected[i_redist_outlier]):
                                    adj_share = total_outlier_redist_val * (weights[i_redist_outlier] / sum_active_weights_outlier)
                                    current_corrected[i_redist_outlier] -= adj_share # Subtract because total_outlier_redist_val is excess
                                    current_adjustments[i_redist_outlier] -= adj_share

                    # --- Scenario 4: Solve Overadjustments ---
                    if scenario_config["solve_overadjust"]:
                        total_overadj_redist_val = 0 # This is the sum of (actual_adj - allowed_adj)
                        components_for_overadj_redist = {0,1,2,3}
                        for i_overadj in range(4): # P, E, R, S
                            prod_name_overadj = lab[i_overadj] + comb_str[i_overadj]
                            uncertainty_val = row_data['E_' + prod_name_overadj]
                            actual_adjustment = current_adjustments[i_overadj]

                            if pd.notna(actual_adjustment) and pd.notna(uncertainty_val):
                                allowed_abs_adj = (1 + OVER_ADJUST_PERC) * uncertainty_val
                                if abs(actual_adjustment) > allowed_abs_adj:
                                    capped_adj = sign(actual_adjustment) * allowed_abs_adj
                                    excess_adj = actual_adjustment - capped_adj # This is the amount "taken back"
                                    
                                    current_corrected[i_overadj] -= excess_adj # Add back to value
                                    current_adjustments[i_overadj] = capped_adj # New adjustment
                                    total_overadj_redist_val += excess_adj # Sum of what was taken back
                                    if i_overadj in components_for_overadj_redist:
                                        components_for_overadj_redist.remove(i_overadj)

                        # Redistribute total_overadj_redist_val (this is sum of excesses, needs to be applied to others)
                        active_weights_overadj = [weights[i] for i in components_for_overadj_redist if pd.notna(weights[i])]
                        sum_active_weights_overadj = sum(active_weights_overadj)
                        if sum_active_weights_overadj != 0 and total_overadj_redist_val != 0:
                            for i_redist_overadj in components_for_overadj_redist:
                                if pd.notna(weights[i_redist_overadj]) and pd.notna(current_corrected[i_redist_overadj]):
                                    adj_share = total_overadj_redist_val * (weights[i_redist_overadj] / sum_active_weights_overadj)
                                    # This redistribution logic is tricky: if P's adjustment was capped, ERS need to compensate
                                    # If total_overadj_redist_val is positive (meaning P's positive adj was capped, or ERS's negative adj was capped)
                                    # For P (i=0), if its adj was capped, others take up the slack.
                                    # If P's adjustment was positive and capped, iter_res is positive. P needs to increase, ERS decrease.
                                    # This needs careful thought on signs. Let's assume adj_share is the amount of budget imbalance.
                                    current_corrected[i_redist_overadj] -= adj_share # If excess was positive, value decreases
                                    current_adjustments[i_redist_overadj] -= adj_share

                    # Store final corrected values for the scenario
                    for i_final in range(4):
                        processed_data_all_scenarios.loc[index, f'{method_name}_{comb_str}_{lab[i_final]}_{scenario_name}'] = current_corrected[i_final]
                        processed_data_all_scenarios.loc[index, f'{method_name}_{comb_str}_{lab[i_final]}_r_{scenario_name}'] = current_adjustments[i_final]
            
            first_msd = False # After the first row, MSD uses cumulative D

    processed_data_all_scenarios.to_csv(outputDir+'synthetic_all_scenarios_results.csv', index=False)
    print(f"Results saved to {outputDir}synthetic_all_scenarios_results.csv")

    # ##### validate with true values ################################################################
    all_scenarios_stats_summary = pd.DataFrame()

    for scenario_config_stats in scenarios:
        scenario_name_stats = scenario_config_stats["name"]
        df_stats_scenario = pd.DataFrame(columns=['Scenario','Component', 'Method', 'Combination', 'PBIAS', 'CC', 'RMSE', 'ME', 'ME1', 'MAE', 'MAPE'])
        
        for method_stat_name in met:
            for comb_stat_str in Combinations:
                for comp_stat_idx, comp_stat_val in enumerate(lab):
                    corrected_col_name = f'{method_stat_name}_{comb_stat_str}_{comp_stat_val}_{scenario_name_stats}'
                    true_col_name = comp_stat_val + '_closed'
                    
                    if corrected_col_name in processed_data_all_scenarios.columns:
                        # Ensure we skip first 3 rows for MSD if it's unstable, or handle NaNs in compute_stats
                        # For simplicity, we'll use all rows here, assuming compute_stats handles NaNs.
                        true_vals = processed_data_all_scenarios[true_col_name]
                        corrected_vals = processed_data_all_scenarios[corrected_col_name]

                        pbias, cc, rmse, me, me1, mae, mape = compute_stats(true_vals, corrected_vals)
                        
                        new_stat_row = pd.DataFrame([{
                            'Scenario': scenario_name_stats, 'Component': comp_stat_val, 'Method': method_stat_name, 
                            'Combination': comb_stat_str, 'PBIAS': pbias, 'CC': cc, 'RMSE': rmse, 
                            'ME': me, 'ME1': me1, 'MAE': mae, 'MAPE': mape
                        }])
                        df_stats_scenario = pd.concat([df_stats_scenario, new_stat_row], ignore_index=True)

        # Aggregate stats for this scenario (mean over combinations)
        agg_stats_scenario = df_stats_scenario.groupby(['Scenario', 'Component', 'Method'])[statistics].mean().reset_index()
        all_scenarios_stats_summary = pd.concat([all_scenarios_stats_summary, agg_stats_scenario], ignore_index=True)

    print("\n--- Aggregated Statistics Summary ---")
    print(all_scenarios_stats_summary.to_string())
    all_scenarios_stats_summary.to_csv(outputDir+'synthetic_aggregated_stats_summary.csv', index=False)

# --- Visualization ---
# Example: Plot RMSE for P component across scenarios for CKF method
plot_component = 'S'
plot_method = 'CKF' # Change as needed, or loop through methods

plt.figure(figsize=(10, 6))
subset_to_plot = all_scenarios_stats_summary[
    (all_scenarios_stats_summary['Component'] == plot_component) &
    (all_scenarios_stats_summary['Method'] == plot_method)
]

if not subset_to_plot.empty:
    sns.barplot(data=subset_to_plot, x='Scenario', y='RMSE', hue='Scenario', dodge=False, palette="viridis")
    plt.title(f'RMSE for {plot_method} - Component {plot_component}')
    plt.ylabel('Mean RMSE')
    plt.xlabel('Scenario')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outputDir+f'RMSE_{plot_method}_{plot_component}_comparison.png')
    plt.show()
else:
    print(f"No data to plot for {plot_method} - {plot_component}")

# Visualize a specific corrected timeseries against true for a chosen scenario/method/combination
chosen_scenario_vis = 'ObsOutlierOver' # Last scenario
chosen_method_vis = 'CKF'
chosen_comb_vis = Combinations[0] # First combination

fig_ts, axes_ts = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig_ts.suptitle(f'Timeseries Comparison: {chosen_method_vis} {chosen_comb_vis} ({chosen_scenario_vis}) vs True Synthetic')

for i_comp_vis, comp_val_vis in enumerate(lab):
    true_data_vis = processed_data_all_scenarios[comp_val_vis + '_closed']
    corrected_data_vis = processed_data_all_scenarios[f'{chosen_method_vis}_{chosen_comb_vis}_{comp_val_vis}_{chosen_scenario_vis}']
    
    axes_ts[i_comp_vis].plot(t_time, true_data_vis, label=f'{comp_val_vis} (True)', color='black', linestyle='-')
    axes_ts[i_comp_vis].plot(t_time, corrected_data_vis, label=f'{comp_val_vis} ({chosen_scenario_vis})', color='red', linestyle='--')
    axes_ts[i_comp_vis].set_ylabel(comp_val_vis)
    axes_ts[i_comp_vis].legend()

axes_ts[-1].set_xlabel('Month')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(outputDir+f'Timeseries_{chosen_method_vis}_{chosen_comb_vis}_{chosen_scenario_vis}.png')
plt.show()