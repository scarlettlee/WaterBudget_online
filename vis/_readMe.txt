结果部分可视化，按照4.1-4.4小节组织数据和图

///////////////////////////// 4.1显示百分比
--两个csv文件分别为3 Basins和28个流域的数据，包含列：
basinID
abnormal：异常情况分类，0表示BCC结果，1表示Outliers（负值和过大的值，即超过原始值2倍的情况），2表示OverlyAdjusted（超过原始值±20%）
method：PR/CKF/MCL/MSD
component:	P/E/R/S
percentage:  负值百分比
--两个图为两类实验的箱图

///////////////////////////// 4.2对比调整Outliers和OverlyAdjusted前后的精度
--目前暂时只实验了3 Basins的数据，包括列：
basin：流域
date：日期
P/E/R/S_closed: 闭合后的参考值
value：相应的值列
method：同上
component：同上
abnormal：同上
combination：观测值组合数字序号
--图
