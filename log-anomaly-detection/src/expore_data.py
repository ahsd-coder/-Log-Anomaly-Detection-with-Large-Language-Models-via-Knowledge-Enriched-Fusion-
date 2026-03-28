import pandas as pd

# 加载结构化数据
structured_df = pd.read_csv('./../data/loghub/BGL/BGL_2k.log_structured.csv')
templates_df = pd.read_csv('./../data/loghub/BGL/BGL_2k.log_templates.csv')

print("结构化数据形状:", structured_df.shape)
print("\n前几行:")
print(structured_df.head())

print("标签分布:")
print(structured_df['Label'].value_counts())

print("模板数据:")
print(templates_df.head())

# 标签二值化（正常 vs 异常）
structured_df['is_anomaly'] = structured_df['Label'] != '-'

# 统计异常比例
anomaly_rate = structured_df['is_anomaly'].mean()
print(f"异常率: {anomaly_rate:.2%}")

# 如果有 EventId 列，可以关联模板
if 'EventId' in structured_df.columns:
    merged_df = structured_df.merge(templates_df, on='EventId', how='left')
    print("关联模板后的数据:")
    print(merged_df.head())

