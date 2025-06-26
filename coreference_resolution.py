from hanlp_restful import HanLPClient
import re
import json

# 使用相同的API配置
API_KEY = 'ODQ1OEBiYnMuaGFubHAuY29tOmlDOE5uTzRCWUFZbllBZEQ='  
HanLP = HanLPClient('https://www.hanlp.com/api', auth=API_KEY, language='zh')

def read_test_file(file_path='test.txt'):
    """读取测试文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        return content
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None

def split_into_sentences(text):
    """将文本分割为句子"""
    # 使用正则表达式按句号、问号、感叹号分割
    sentences = re.split(r'[。！？]+', text)
    
    # 过滤空句子并去除前后空格
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def get_coreference_resolution(sentence):
    """获取句子的指代消解结果"""
    try:
        result = HanLP.coreference_resolution(sentence)
        return result.get('clusters', [])
    except Exception as e:
        print(f"指代消解出错: {e}")
        return []

def build_coreference_mapping(clusters):
    """构建指代消解映射表"""
    mapping = {}
    for cluster in clusters:
        if len(cluster) >= 2:  # 至少有两个指代关系
            # 找到最长的实体作为代表实体（通常是最具体的实体）
            representative = max(cluster, key=lambda x: len(x[0]))
            representative_entity = representative[0]
            
            # 将其他实体映射到代表实体
            for entity, start, end in cluster:
                if entity != representative_entity:
                    mapping[entity] = representative_entity
    
    return mapping

def resolve_coreference_in_triples(triples, coreference_mapping):
    """解决三元组中的指代问题"""
    if not coreference_mapping:
        return triples
    
    resolved_triples = []
    
    for subject, predicate, obj in triples:
        # 解决主语中的指代
        resolved_subject = coreference_mapping.get(subject, subject)
        
        # 解决宾语中的指代
        resolved_object = coreference_mapping.get(obj, obj)
        
        # 检查主语和宾语中是否包含需要替换的指代词
        for pronoun, entity in coreference_mapping.items():
            if pronoun in resolved_subject:
                resolved_subject = resolved_subject.replace(pronoun, entity)
            if pronoun in resolved_object:
                resolved_object = resolved_object.replace(pronoun, entity)
        
        resolved_triples.append((resolved_subject, predicate, resolved_object))
    
    return resolved_triples

def get_sentence_coreference_info(sentence):
    """获取单个句子的完整指代消解信息"""
    try:
        result = HanLP.coreference_resolution(sentence)
        clusters = result.get('clusters', [])
        mapping = build_coreference_mapping(clusters)
        
        return {
            'clusters': clusters,
            'mapping': mapping,
            'tokens': result.get('tokens', [])
        }
    except Exception as e:
        print(f"获取指代消解信息出错: {e}")
        return {
            'clusters': [],
            'mapping': {},
            'tokens': []
        }

class CompactJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于生成紧凑格式的JSON"""
    def encode(self, obj):
        if isinstance(obj, dict):
            # 对于字典，检查是否包含特定键
            if any(key in obj for key in ['clusters', 'mapping', 'tokens']):
                # 为这些特定字段创建单行格式
                items = []
                for key, value in obj.items():
                    if key in ['clusters', 'mapping', 'tokens']:
                        # 将这些字段的值压缩到一行
                        value_str = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                        items.append(f'"{key}": {value_str}')
                    else:
                        # 其他字段正常处理
                        value_str = json.dumps(value, ensure_ascii=False)
                        items.append(f'"{key}": {value_str}')
                return '{' + ', '.join(items) + '}'
        
        # 默认处理
        return super().encode(obj)

def save_coreference_results(results, output_file='coreference_analysis.json'):
    """保存指代消解结果到文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 使用自定义的紧凑格式
            f.write('[\n')
            for i, result in enumerate(results):
                if i > 0:
                    f.write(',\n')
                
                # 手动格式化每个结果项
                f.write('  {\n')
                f.write(f'    "sentence_num": {result["sentence_num"]},\n')
                f.write(f'    "sentence": {json.dumps(result["sentence"], ensure_ascii=False)},\n')
                f.write('    "coreference_info": {\n')
                
                coreference_info = result["coreference_info"]
                
                # clusters 压缩到一行
                clusters_str = json.dumps(coreference_info["clusters"], ensure_ascii=False, separators=(',', ':'))
                f.write(f'      "clusters": {clusters_str},\n')
                
                # mapping 压缩到一行
                mapping_str = json.dumps(coreference_info["mapping"], ensure_ascii=False, separators=(',', ':'))
                f.write(f'      "mapping": {mapping_str},\n')
                
                # tokens 压缩到一行
                tokens_str = json.dumps(coreference_info["tokens"], ensure_ascii=False, separators=(',', ':'))
                f.write(f'      "tokens": {tokens_str}\n')
                
                f.write('    }\n')
                f.write('  }')
            f.write('\n]')
        
        print(f"指代消解结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存指代消解结果出错: {e}")

def load_coreference_results(input_file='coreference_analysis.json'):
    """加载指代消解结果"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return None
    except Exception as e:
        print(f"加载指代消解结果出错: {e}")
        return None

# 主程序
if __name__ == "__main__":
    # 读取测试文件内容
    text = read_test_file('test.txt')
    
    if text is None:
        print("无法读取测试文件，程序退出")
        exit(1)
    
    # 分割为句子
    sentences = split_into_sentences(text)
    print(f"共分割出 {len(sentences)} 个句子")
    
    # 存储所有句子的指代消解结果
    all_coreference_results = []
    
    # 逐句处理
    for i, sentence in enumerate(sentences, 1):
        if sentence:  # 确保句子不为空
            print(f"\n句子{i}: {sentence}")
            
            # 获取指代消解信息
            coreference_info = get_sentence_coreference_info(sentence)
            
            # 显示结果
            if coreference_info['clusters']:
                print(f"指代消解簇: {coreference_info['clusters']}")
                print(f"映射关系: {coreference_info['mapping']}")
            else:
                print("无指代关系")
            
            # 保存结果
            all_coreference_results.append({
                'sentence_num': i,
                'sentence': sentence,
                'coreference_info': coreference_info
            })
    
    # 保存所有结果到文件
    save_coreference_results(all_coreference_results)
    
    print(f"\n指代消解分析完成，共处理 {len(sentences)} 个句子")