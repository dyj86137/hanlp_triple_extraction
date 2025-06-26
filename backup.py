import hanlp
from hanlp_restful import HanLPClient
import re
from itertools import combinations

# HanLP API 
HanLP = HanLPClient('https://www.hanlp.com/api', auth=None, language='zh')

def filter_contained_keyphrases(keyphrases_dict):
    """
    过滤掉被其他关键词短语包含的短语
    """
    phrases = list(keyphrases_dict.keys())
    filtered_dict = {}

    for phrase in phrases:
        # 检查当前短语是否被其他短语包含
        is_contained = False
        for other_phrase in phrases:
            if phrase != other_phrase and phrase in other_phrase:
                is_contained = True
                break
        
        # 如果没有被包含，则保留
        if not is_contained:
            filtered_dict[phrase] = keyphrases_dict[phrase]
    
    return filtered_dict

def extract_relations(sentence, entities):
    # 获取句法依存分析结果
    dep = HanLP.parse(sentence, tasks='dep')
    triples = []
    
    # 实体已经按照在句子中出现的顺序排序
    for e1, e2 in combinations(entities, 2):
        if e1 in sentence and e2 in sentence:
            # 获取实体在句子中的实际位置
            idx1, idx2 = sentence.find(e1), sentence.find(e2)
            
            # 如果需要，交换顺序确保idx1在前
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
                e1, e2 = e2, e1
            
            # 提取两个实体之间的文本作为关系
            relation_span = sentence[idx1 + len(e1):idx2] if idx2 > idx1 + len(e1) else ""
            relation = relation_span.strip()
            
            # 确保关系非空且有意义
            if relation and len(relation) <= 20:  # 限制关系长度，避免太长的无意义文本
                # 清理关系文本
                relation = re.sub(r'[，。、；：！？\s]+', '', relation)
                if relation:
                    triples.append((e1, relation, e2))
    
    return triples

# 读取文本
with open('test.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 按句号分句，过滤掉空句子
sentences = [s.strip() for s in text.split('。') if s.strip()]

print(f"共分割出 {len(sentences)} 个句子")
print("=" * 50)

# 对每个句子进行关键词短语提取和关系抽取
for i, sentence in enumerate(sentences, 1):
    print(f"句子 {i}: {sentence}")
    try:
        keyphrases = HanLP.keyphrase_extraction(sentence, topk=10)
        filtered_keyphrases = filter_contained_keyphrases(keyphrases)
        entities = list(filtered_keyphrases.keys())
        
        # 按照实体在句子中出现的顺序排序
        entities_with_position = []
        for entity in entities:
            pos = sentence.find(entity)
            if pos != -1:  # 确保实体在句子中
                entities_with_position.append((entity, pos))
        
        # 按位置排序
        entities_with_position.sort(key=lambda x: x[1])
        
        # 提取排序后的实体列表
        sorted_entities = [entity for entity, pos in entities_with_position]
        
        print(f"实体 (按出现顺序排序): {sorted_entities}")

        triples = extract_relations(sentence, sorted_entities)
        print(f"三元组: {triples}")

    except Exception as e:
        print(f"处理错误: {e}")
    print("-" * 30)