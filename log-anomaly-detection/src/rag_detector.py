import pandas as pd 
import numpy as np  #数据处理基础库，用于处理日志数据的表格结构
from sentence_transformers import SentenceTransformer #句子向量化模型，将文本（日志模板 / 日志内容）转换成数值向量
import faiss #Facebook 开源的向量检索库，高效检索相似向量
from openai import OpenAI #OpenAI 的 Python 客户端，调用 GPT 模型做异常判断
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix #评估指标（精准率、召回率等），虽然代码里没用到，但预留了评估检测效果的能力
import os #读取环境变量（比如 OpenAI 的 API Key）

class LogAnomalyRAG:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        """初始化 RAG 日志异常检测器"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.index = None
        self.knowledge_base = []
        
    def build_knowledge_base(self, templates_df, structured_df):
        """构建知识库"""
        print("🔨 构建知识库...")
        
        # 1. 从模板构建基础知识
        for idx, row in templates_df.iterrows():
            event_id = row['EventId']
            template = row['EventTemplate']
            
            # 获取该模板对应的历史日志
            template_logs = structured_df[structured_df['EventId'] == event_id]
            
            # 统计正常/异常比例
            total_count = len(template_logs)
            anomaly_count = template_logs['is_anomaly'].sum()
            normal_count = total_count - anomaly_count
            
            knowledge_entry = {
                'event_id': event_id,
                'template': template,
                'total_count': total_count,
                'normal_count': normal_count,
                'anomaly_count': anomaly_count,
                'anomaly_rate': anomaly_count / total_count if total_count > 0 else 0,
                'description': f"Template: {template} | Historical stats: {normal_count} normal, {anomaly_count} anomalies"
            }
            self.knowledge_base.append(knowledge_entry)
        
        # 2. 向量化知识库
        knowledge_texts = [kb['description'] for kb in self.knowledge_base]
        knowledge_vectors = self.embedding_model.encode(knowledge_texts, show_progress_bar=True)
        
        # 3. 创建 FAISS 索引
        dimension = knowledge_vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(knowledge_vectors).astype('float32'))
        
        print(f"✅ 知识库构建完成: {len(self.knowledge_base)} 个模板")
        
    def retrieve_knowledge(self, query_text, k=3):
        """检索相关知识"""
        query_vector = self.embedding_model.encode([query_text])
        D, I = self.index.search(np.array(query_vector).astype('float32'), k)
        
        retrieved = []
        for idx in I[0]:
            retrieved.append(self.knowledge_base[idx])
        
        return retrieved
    
    def detect_one_step(self, log_entry, k=3):
        """One-step RAG 检测"""
        # 检索相关知识
        retrieved_knowledge = self.retrieve_knowledge(log_entry, k)
        
        # 构建上下文
        context = "\n".join([
            f"- {kb['description']} (Anomaly rate: {kb['anomaly_rate']:.1%})"
            for kb in retrieved_knowledge
        ])
        
        # 构建提示词
        prompt = f"""You are a log anomaly detection expert.

Analyze this log entry:
