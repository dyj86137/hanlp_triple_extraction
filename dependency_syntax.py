from hanlp_restful import HanLPClient
import re
import json

# 使用官方API
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

def analyze_single_sentence(sentence, sentence_num):
    """分析单个句子的句法依存关系"""
    try:
        # 进行多任务分析
        doc = HanLP.parse(sentence, 
                          tasks=['tok/coarse', 'pos/ctb', 'dep'], skip_tasks='tok/fine')
        
        tokens = doc.get('tok/coarse', [[]])[0]
        pos_tags = doc.get('pos/ctb', [[]])[0]
        dependencies = doc.get('dep', [[]])[0]
        
        # 生成调试信息
        debug_info = []
        for i, (token, pos, dep) in enumerate(zip(tokens, pos_tags, dependencies)):
            debug_text = f"索引{i}: {token} ({pos}) -> 头节点{dep[0]}, 关系{dep[1]}"
            debug_info.append(debug_text)
        
        # 找到root节点
        root_idx = -1
        root_info = ""
        for i, (head, relation) in enumerate(dependencies):
            if relation == 'root':
                root_idx = i
                break
        
        if root_idx >= 0:
            root_info = f"Root节点: {tokens[root_idx]} (索引{root_idx})"
        else:
            root_info = "Root节点: 未找到"
        
        # 返回分析结果（包含简化的调试信息）
        return {
            'sentence': sentence,
            'tokens': tokens,
            'pos_tags': pos_tags,
            'dependencies': dependencies,
            'root_idx': root_idx,
            'root_info': root_info,
            'debug_info': debug_info,
            'word_count': len(tokens)
        }
        
    except Exception as e:
        print(f"句子{sentence_num}分析出错: {e}")
        return None

def save_analysis_results(results, output_file='dependency_analysis.json'):
    """保存分析结果到JSON文件"""
    try:
        import json
        
        # 自定义JSON序列化
        def custom_json_format(obj, indent=0):
            if isinstance(obj, dict):
                if not obj:
                    return "{}"
                
                lines = ["{"]
                items = list(obj.items())
                for i, (key, value) in enumerate(items):
                    comma = "," if i < len(items) - 1 else ""
                    
                    # 对于数组类型的值，保持在一行
                    if isinstance(value, list):
                        value_str = json.dumps(value, ensure_ascii=False)
                    else:
                        value_str = custom_json_format(value, indent + 1)
                    
                    lines.append(f"{'  ' * (indent + 1)}\"{key}\": {value_str}{comma}")
                
                lines.append(f"{'  ' * indent}}}")
                return "\n".join(lines)
            
            elif isinstance(obj, list):
                if not obj:
                    return "[]"
                
                lines = ["["]
                for i, item in enumerate(obj):
                    comma = "," if i < len(obj) - 1 else ""
                    item_str = custom_json_format(item, indent + 1)
                    lines.append(f"{'  ' * (indent + 1)}{item_str}{comma}")
                
                lines.append(f"{'  ' * indent}]")
                return "\n".join(lines)
            
            else:
                return json.dumps(obj, ensure_ascii=False)
        
        # 生成自定义格式的JSON
        json_content = custom_json_format(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        print(f"\n分析结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果出错: {e}")

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
    
    # 存储所有句子的分析结果
    all_results = []
    
    # 逐句分析
    for i, sentence in enumerate(sentences, 1):
        if sentence:  # 确保句子不为空
            result = analyze_single_sentence(sentence, i)
            if result:
                all_results.append({
                    'sentence_num': i,
                    'analysis': result
                })
    
    # 保存分析结果
    save_analysis_results(all_results)