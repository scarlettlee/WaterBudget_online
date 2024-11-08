# A test program for water cycle

## Steps to compute: 
1. BCC: Close water balance components with budget component closure methods PR, CKF, MCL, and MSD
2. closeMergedComponents: Compute closed components as reference values
3. redistribute: Compute outliers and overlyAdjusted values, and then redistribute the residual
4. compareMethods_stats: Compute statistical indicators RMSE, Correlation Coefficient (CC), PBIAS, etc.

## Introduction of all implementation python files:
File | Function | Input | Output folder
--- | --- | --- | ---
globVar.py | Define global variables and functions | - | -
BCCs_introduceObs.py | BCC methods with observations introduced as reference data | *stationsPrecipitation.xlsx* and *data_basin* | *BasinsComparison_obsIntroduced*
CloseMergedComponents_partTrue.py | Compute closed components as reference values | *stationsPrecipitation.xlsx* and *BasinsComparison_obsIntroduced* | *BasinsComparison_mergeClosed_partTrue*
Redistribute_mergeClosed_refClosed.py | Compute outliers and overlyAdjusted values, and then redistribute the residual | *BasinsComparison_mergeClosed_partTrue* | *redistribution_outliers_mergeClosed_partTrue*
CompareMethods_..._mergeClosed.py | Compute statistical indicators RMSE, Correlation Coefficient (CC), PBIAS, etc. | *redistribution_outliers_mergeClosed_partTrue* | *stats_mergedClosed_partTrue*
`pics 4 debug`, `output_test`, `results visualization`,`test`,`_old files and folders` | assistant folders | - | -

## Introduction of visualization python files for results:
File | Results Section | Input | Output
--- | --- | --- | ---
vis1_percentage.py | 4.1 Percentage of ourliers and overlyAdjusted values for each component | *redistribution_outliers_mergeClosed_partTrue* | *results visualization*
vis2_accuracyImprove.py | 4.2 Accuracy comparison based on our proposed method | *redistribution_outliers_mergeClosed_partTrue* | *results visualization*
vis4_synthetic.py | 4.4 Synthetic experiments | - | -