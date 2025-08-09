# 水循环测试程序 - 详细技术说明

## 项目概述
这是一个用于水循环分量闭合和异常值检测的Python程序包，主要实现了多种预算分量闭合方法（BCC）以及后续的异常值检测和重新分配算法。

## 核心算法原理

### 1. 预算分量闭合方法（BCC）
程序实现了四种主要的BCC方法：

- **PR (Product Ratio)**: 乘积比率方法
- **CKF (Constrained Kalman Filter)**: 约束卡尔曼滤波方法  
- **MCL (Minimum Constraint Least squares)**: 最小约束最小二乘法
- **MSD (Minimum Squared Distance)**: 最小平方距离方法

这些方法用于解决水平衡方程 P - E - R - S = 0 的闭合问题，其中：
- P: 降水量 (Precipitation)
- E: 蒸散发量 (Evapotranspiration)  
- R: 径流量 (Runoff)
- S: 陆地水储量变化 (Terrestrial Water Storage Change)

### 2. 异常值检测算法
程序采用两阶段异常值检测策略：

- **第一阶段 (r1)**: 检测符号异常值
  - 当参考值与BCC结果符号相反时，标记为异常
  - 例如：参考P为负值，但BCC结果P为正值

- **第二阶段 (r2)**: 检测过度调整值
  - 当BCC结果超过参考值一定比例时（默认50%），标记为过度调整
  - 可配置参数：`overAdjustPerc = 0.5`

### 3. 重新分配算法
异常值检测后，程序会：
1. 计算异常值数量（r1和r2）
2. 将异常值重新分配回原始分量
3. 保持水平衡方程的闭合性

## 计算流程详解

### 步骤1: BCC计算 (`BCCs_introduceObs.py`)
```python
# 主要功能：
- 读取多源遥感数据（降水、蒸散发、径流、陆地水储量）
- 应用四种BCC方法进行分量闭合
- 引入观测数据作为参考值
- 输出闭合后的分量数据
```

**输入数据格式**：
- 降水数据：GPCC、GPCP、Gsmap、IMERG、PERSIANN_CDR等
- 蒸散发数据：FLUXCOM、GLDAS、GLEAM、PT-JPL等  
- 径流数据：GRDC观测站数据
- 陆地水储量：GRACE CSR、GFZ、JPL、Mascon等

### 步骤2: 闭合分量计算 (`CloseMergedComponents_partTrue.py`)
```python
# 主要功能：
- 使用观测降水数据替换BCC结果中的降水值
- 重新计算蒸散发：E_closed = P_closed - R_closed - S_closed
- 确保水平衡方程的物理一致性
```

### 步骤3: 异常值检测与重新分配 (`Redistribute_mergeClosed_refClosed.py`)
```python
# 核心算法：
def compute_newR1(row, l, col):
    # 检测符号异常值
    if (a < 0 and b > 0) | (a > 0 and b < 0):
        row[col+'_r1'] = b-a  # 计算异常值大小
        
def compute_newR2(row, l, col):
    # 检测过度调整值
    if abs(b) > (1+overAdjustPerc)*a:
        row[col+'2'] = b - sign(b)*(1+overAdjustPerc)*a
```

### 步骤4: 统计评估 (`CompareMethods_stats_redistributed_mergeClosed.py`)
程序计算多种统计指标：
- **RMSE**: 均方根误差
- **CC**: 相关系数  
- **PBIAS**: 百分比偏差
- **ME**: 平均误差
- **MAE**: 平均绝对误差
- **MAPE**: 平均绝对百分比误差

## 数据组织结构

### 输入数据文件夹
- `3data_basin/`: 3个测试流域的原始数据
- `28data_basin/`: 28个流域的原始数据
- `stationsPrecipitation.xlsx`: 观测降水数据

### 输出数据文件夹
- `3BasinsComparison_obsIntroduced/`: BCC结果
- `3BasinsComparison_mergeClosed_partTrue/`: 闭合分量结果
- `3redistribution_outliers_mergeClosed_partTrue/`: 异常值检测结果
- `3stats_mergedClosed_partTrue/`: 统计评估结果

### 数据文件格式
CSV文件包含以下列：
- 时间列：`Unnamed: 0` (格式：YYYYMM)
- 降水列：`Pre_GPCC`, `Pre_GPCP`, `Pre_Gsmap`等
- 蒸散发列：`ET_FLUXCOM`, `ET_GLDAS`, `ET_GLEAM`等
- 径流列：`GRDC`
- 陆地水储量列：`TWSC_GRACE_CSR_calculate`等

## 可视化功能

### 1. 异常值百分比分析 (`vis1_percentage.py`)
- 分析各分量异常值（r1、r2）的百分比
- 生成4×2的子图矩阵，展示P、E、R、S各分量的异常情况
- 支持3流域和28流域两种模式

### 2. 精度改进分析 (`vis2_accuracyImprove.py`)
- 比较不同BCC方法的精度改进效果
- 分析异常值重新分配前后的精度变化

### 3. 合成实验 (`vis4_synthetic.py`)
- 使用正弦波生成合成数据
- 验证算法在不同数据特征下的表现
- 支持参数化实验设计

## 全局配置与工具函数

### 核心配置 (`globVar.py`)
```python
# 主要配置参数
basin3Flag = True  # True: 3流域模式, False: 28流域模式
strategy = "mergeAsRef"  # 参考值合并策略

# 核心工具函数
def compute_stats(true_values, corrected_values):  # 统计指标计算
def getMergedTrue(arr):  # 参考值合并
def generateAInverse(paramNum):  # MCL方法系数矩阵生成
def getSquaredDistance(x,y):  # MSD方法距离计算
def getUncertCoefR(x,index):  # 径流不确定性系数
```

### 文件操作工具
- `find_pattern()`: 文件模式匹配
- `get_file_name()`: 文件名提取
- 支持通配符和正则表达式匹配

## 使用方法

### 1. 环境准备
```bash
# 安装必要的Python包
pip install pandas numpy matplotlib seaborn openpyxl
```

### 2. 数据准备
- 将流域数据放入对应的`*data_basin/`文件夹
- 确保观测降水数据`stationsPrecipitation.xlsx`存在
- 检查数据格式和列名是否匹配

### 3. 运行顺序
```bash
# 1. 运行BCC计算
python BCCs_introduceObs.py

# 2. 计算闭合分量
python CloseMergedComponents_partTrue.py

# 3. 异常值检测与重新分配
python Redistribute_mergeClosed_refClosed.py

# 4. 统计评估
python CompareMethods_stats_redistributed_mergeClosed.py

# 5. 结果可视化
python vis1_percentage.py
python vis2_accuracyImprove.py
```

### 4. 参数调整
- 修改`globVar.py`中的`basin3Flag`切换流域数量
- 调整`overAdjustPerc`参数控制过度调整阈值
- 修改`test`标志进行单文件测试

## 技术特点

### 1. 算法优势
- **多方法集成**: 支持四种主流BCC方法
- **异常值检测**: 两阶段检测策略，提高可靠性
- **物理一致性**: 确保水平衡方程的闭合性
- **观测融合**: 引入地面观测数据提高精度

### 2. 代码设计
- **模块化结构**: 功能分离，便于维护和扩展
- **配置集中**: 全局参数统一管理
- **错误处理**: 完善的异常值处理机制
- **可视化支持**: 多种图表展示结果

### 3. 应用场景
- 流域水平衡分析
- 遥感数据质量评估
- 水文模型验证
- 气候变化研究

## 注意事项

### 1. 数据要求
- 时间序列数据需要连续且格式一致
- 缺失值处理：程序会自动跳过NaN值
- 数据单位：建议使用mm/month或mm/day

### 2. 计算资源
- 28流域模式需要更多内存和计算时间
- 建议先使用3流域模式测试功能
- 大数据集处理时注意内存使用

### 3. 结果解释
- 异常值检测结果需要结合物理意义判断
- 统计指标需要结合具体应用场景评估
- 建议进行敏感性分析验证参数设置

## 扩展开发

### 1. 新增BCC方法
在`globVar.py`中添加新的算法实现，并在主程序中调用

### 2. 自定义异常值检测
修改`compute_newR1`和`compute_newR2`函数，实现特定的检测逻辑

### 3. 新增统计指标
在`compute_stats`函数中添加新的评估指标

### 4. 数据源扩展
支持更多遥感数据产品和观测数据源

## 常见问题

### Q1: 程序运行出错怎么办？
- 检查数据文件路径和格式
- 确认Python包版本兼容性
- 查看错误信息定位具体问题

### Q2: 如何理解异常值检测结果？
- r1表示符号异常，需要重点关注
- r2表示过度调整，可根据应用需求调整阈值
- 结合物理意义判断异常值的合理性

### Q3: 如何选择合适的BCC方法？
- 根据数据质量和应用需求选择
- 建议对比多种方法的结果
- 考虑计算复杂度和精度要求


