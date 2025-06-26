from hanlp_restful import HanLPClient
from keyword_extraction import filter_contained_keyphrases
from coreference_resolution import get_sentence_coreference_info, resolve_coreference_in_triples
import json
import re

# 使用官方API
API_KEY = 'ODQ1OEBiYnMuaGFubHAuY29tOmlDOE5uTzRCWUFZbllBZEQ='  
HanLP = HanLPClient('https://www.hanlp.com/api', auth=API_KEY, language='zh')

def load_dependency_analysis(input_file='dependency_analysis.json'):
    """加载句法依存分析结果"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"已加载句法分析结果: {input_file}")
        return results
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return None
    except Exception as e:
        print(f"加载分析结果出错: {e}")
        return None

def get_filtered_keyphrases(sentence):
    """获取过滤后的关键词短语"""
    try:
        keyphrases = HanLP.keyphrase_extraction(sentence, topk=15)
        filtered_keyphrases = filter_contained_keyphrases(keyphrases)
        return list(filtered_keyphrases.keys())
    except Exception as e:
        print(f"关键词提取出错: {e}")
        return []

def extract_triples(tokens, pos_tags, dependencies):
    """基于依存句法提取三元组"""
    triples = []
    
    # 处理所有动词的三元组关系
    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        if pos in ['VV', 'VC']:  # 找到所有动词
            verb = token
            verb_subjects = []
            verb_objects = []
            
            # 找到这个动词的主语和宾语
            for j, (head, relation) in enumerate(dependencies):
                if head - 1 == i:  # 依存于当前动词的词
                    if relation == 'nsubj':
                        verb_subjects.append(tokens[j])
                    elif relation == 'dobj':
                        verb_objects.append(tokens[j])
            
            # 如果是系动词，特殊处理
            if pos == 'VC':
                # 找到系动词连接的主语和宾语
                cop_head_idx = -1
                for j, (head, relation) in enumerate(dependencies):
                    if j == i and relation == 'cop':
                        cop_head_idx = head - 1  # 系动词指向的宾语
                        break
                
                if cop_head_idx >= 0:
                    # 找这个宾语的主语
                    for j, (head, relation) in enumerate(dependencies):
                        if head - 1 == cop_head_idx and relation == 'nsubj':
                            subject = tokens[j]
                            predicate = tokens[cop_head_idx]  # 宾语作为谓语
                            triples.append((subject, verb, predicate))
            
            # 处理一般动词
            elif pos == 'VV':
                # 如果没有明确主语，需要推断
                if not verb_subjects and verb_objects:
                    # 推断主语：可能是句子的主语或前面提到的实体
                    inferred_subjects = []
                    
                    # 1. 如果是并列动词，主语可能与前面的动词共享
                    for j, (head, relation) in enumerate(dependencies):
                        if j == i and relation == 'conj':
                            # 找前面动词的主语
                            prev_verb_idx = head - 1
                            if prev_verb_idx >= 0:
                                # 查找前面动词的宾语作为当前动词的主语
                                for k, (h, rel) in enumerate(dependencies):
                                    if h - 1 == prev_verb_idx and rel == 'dobj':
                                        inferred_subjects.append(tokens[k])
                    
                    # 2. 如果还没找到，查找句子中的主语
                    if not inferred_subjects:
                        for j, (head, relation) in enumerate(dependencies):
                            if relation == 'nsubj':  # 任何主语关系
                                inferred_subjects.append(tokens[j])
                    
                    # 构建三元组
                    for subj in inferred_subjects:
                        for obj in verb_objects:
                            triples.append((subj, verb, obj))
                
                # 如果有明确的主语和宾语
                elif verb_subjects and verb_objects:
                    for subj in verb_subjects:
                        for obj in verb_objects:
                            triples.append((subj, verb, obj))
    
    return list(set(triples))  # 去重

def find_keyphrase_positions(keyphrase, tokens):
    """找到关键短语在句子中的最佳位置"""
    positions = []
    
    # 方法1: 直接完整匹配优先
    for i, token in enumerate(tokens):
        if token == keyphrase:
            return [i]  # 完全匹配，直接返回
    
    # 方法2: 查找关键短语的起始位置
    kp_tokens = tokenize_keyphrase(keyphrase)
    if len(kp_tokens) > 1:
        # 多词关键短语，找连续匹配的起始位置
        for i in range(len(tokens) - len(kp_tokens) + 1):
            match_count = 0
            for j, kp_token in enumerate(kp_tokens):
                if i + j < len(tokens) and tokens_similar(tokens[i + j], kp_token):
                    match_count += 1
                else:
                    break
            
            # 如果匹配度足够高，记录起始位置
            if match_count >= max(1, len(kp_tokens) * 0.6):
                positions.append(i)
    else:
        # 单词关键短语，找所有匹配位置
        kp_token = kp_tokens[0] if kp_tokens else keyphrase
        for i, token in enumerate(tokens):
            if tokens_similar(token, kp_token):
                positions.append(i)
    
    # 方法3: 如果还没找到，使用包含关系
    if not positions:
        for i, token in enumerate(tokens):
            if token in keyphrase or keyphrase in token:
                positions.append(i)
    
    return positions

def find_best_keyphrase_match_with_context(word, word_index, keyphrases, tokens):
    """根据上下文位置找到最佳匹配的关键词短语"""
    print(f"  匹配词汇: {word} (索引: {word_index})")
    
    # 1. 精确匹配
    if word in keyphrases:
        print(f"  精确匹配: {word}")
        return word
    
    # 2. 根据位置找最近的包含该词的关键短语
    candidates = []
    for kp in keyphrases:
        if word in kp:
            # 找到关键短语的最佳位置
            kp_positions = find_keyphrase_positions(kp, tokens)
            
            if kp_positions:
                # 选择距离目标词最近的位置
                best_kp_position = min(kp_positions, key=lambda pos: abs(word_index - pos))
                distance = abs(word_index - best_kp_position)
                candidates.append((kp, distance, best_kp_position))
                print(f"  候选: {kp}, 距离: {distance}, 最佳位置: {best_kp_position}")
    
    if candidates:
        # 选择距离最近的关键短语，如果距离相同则选择更具体的（更长的）
        best_match = min(candidates, key=lambda x: (x[1], -len(x[0])))
        print(f"  最佳匹配: {best_match[0]}")
        return best_match[0]
    
    # 3. 找该词包含的最长关键短语
    candidates = [kp for kp in keyphrases if kp in word]
    if candidates:
        result = max(candidates, key=len)
        print(f"  包含匹配: {result}")
        return result
    
    # 4. 如果都没找到，返回原词
    print(f"  无匹配，返回原词: {word}")
    return word

def enhance_triples_with_keyphrases_contextual(triples, keyphrases, tokens, pos_tags, dependencies):
    """用关键词短语增强三元组（考虑上下文）"""
    enhanced_triples = []
    
    print("增强三元组详细过程:")
    
    for subject, predicate, obj in triples:
        print(f"\n处理三元组: ({subject}, {predicate}, {obj})")
        
        # 找到主体和客体在原句中的正确位置
        subject_index = -1
        object_index = -1
        
        # 首先找到谓语动词的位置
        predicate_index = -1
        for i, token in enumerate(tokens):
            if token == predicate:
                predicate_index = i
                break
        
        print(f"谓语 '{predicate}' 位置: {predicate_index}")
        
        # 通过依存关系确定主体的正确位置
        for i, (head, relation) in enumerate(dependencies):
            if relation == 'nsubj' and head - 1 == predicate_index and tokens[i] == subject:
                subject_index = i
                print(f"找到主语 '{subject}' 位置: {subject_index}")
                break
        
        # 通过依存关系确定客体的正确位置  
        for i, (head, relation) in enumerate(dependencies):
            if relation == 'dobj' and head - 1 == predicate_index and tokens[i] == obj:
                object_index = i
                print(f"找到宾语 '{obj}' 位置: {object_index}")
                break
        
        # 如果通过依存关系没找到，使用简单搜索
        if subject_index == -1:
            for i, token in enumerate(tokens):
                if token == subject:
                    subject_index = i
                    print(f"简单匹配主语 '{subject}' 位置: {subject_index}")
                    break
        
        if object_index == -1:
            for i, token in enumerate(tokens):
                if token == obj:
                    object_index = i
                    print(f"简单匹配宾语 '{obj}' 位置: {object_index}")
                    break
        
        # 根据位置找到最佳匹配
        enhanced_subject = find_best_keyphrase_match_with_context(
            subject, subject_index, keyphrases, tokens)
        enhanced_object = find_best_keyphrase_match_with_context(
            obj, object_index, keyphrases, tokens)
        
        enhanced_triples.append((enhanced_subject, predicate, enhanced_object))
        print(f"增强结果: ({enhanced_subject}, {predicate}, {enhanced_object})")
    
    return enhanced_triples

def sort_triples_by_subject_order(triples, tokens):
    """根据主体在句子中的出现顺序对三元组排序"""
    def get_subject_position(triple):
        subject = triple[0]
        # 找到主体在句子中的最早出现位置
        for i, token in enumerate(tokens):
            if token == subject or token in subject or subject in token:
                return i
        # 如果没找到精确匹配，尝试找部分匹配
        for i, token in enumerate(tokens):
            if any(word in token for word in subject.split()) or any(word in subject for word in token.split()):
                return i
        return len(tokens)  # 如果都没找到，放到最后
    
    return sorted(triples, key=get_subject_position)

def tokenize_keyphrase(keyphrase):
    """分词函数"""
    # 按字符类型分割：字母数字、中文、其他
    tokens = []
    current_token = ""
    current_type = None
    
    for char in keyphrase:
        char_type = get_char_type(char)
        
        if char_type != current_type:
            if current_token.strip():
                tokens.append(current_token)
            current_token = char
            current_type = char_type
        else:
            current_token += char
    
    if current_token.strip():
        tokens.append(current_token)
    
    # 进一步分割长词
    final_tokens = []
    for token in tokens:
        if len(token) > 3 and get_char_type(token[0]) == 'chinese':
            # 将长中文词按字符分割
            for i in range(0, len(token), 2):
                sub_token = token[i:i+2] if i+1 < len(token) else token[i]
                final_tokens.append(sub_token)
        else:
            final_tokens.append(token)
    
    return [t for t in final_tokens if t.strip()]

def get_char_type(char):
    """获取字符类型"""
    if re.match(r'[A-Za-z0-9\-]', char):
        return 'alphanumeric'
    elif re.match(r'[\u4e00-\u9fff]', char):
        return 'chinese'
    else:
        return 'other'

def tokens_similar(token1, token2, threshold=0.6):
    """判断两个词汇是否相似"""
    if token1 == token2:
        return True
    
    if token1 in token2 or token2 in token1:
        return True
    
    # 字符重叠度
    if len(token1) > 0 and len(token2) > 0:
        common = set(token1) & set(token2)
        similarity = len(common) / max(len(set(token1)), len(set(token2)))
        return similarity >= threshold
    
    return False

def save_triples_to_json(clean_results, output_file='triples.json'):
    """保存三元组结果到JSON文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
            
            sentence_nums = sorted(clean_results.keys())
            for i, sentence_num in enumerate(sentence_nums):
                if i > 0:
                    f.write(',\n')
                
                data = clean_results[sentence_num]
                
                # 手动格式化每个句子的结果
                f.write('  {\n')
                f.write(f'    "sentence_num": {sentence_num},\n')
                f.write(f'    "sentence": {json.dumps(data["text"], ensure_ascii=False)},\n')
                f.write(f'    "triple_count": {len(data["triples"])},\n')
                
                # 将三元组数组压缩到一行
                triples_list = []
                for triple in data["triples"]:
                    subject, predicate, obj = triple
                    triples_list.append([subject, predicate, obj])
                
                triples_str = json.dumps(triples_list, ensure_ascii=False, separators=(',', ':'))
                f.write(f'    "triples": {triples_str}\n')
                
                f.write('  }')
            
            f.write('\n]')
        
        print(f"三元组结果已保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"保存三元组结果出错: {e}")
        return False

def load_triples_from_json(input_file='triples.json'):
    """从JSON文件加载三元组结果"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return None
    except Exception as e:
        print(f"加载三元组结果出错: {e}")
        return None

def generate_summary_statistics(clean_results):
    """生成三元组提取的统计摘要"""
    total_sentences = len(clean_results)
    total_triples = sum(len(data['triples']) for data in clean_results.values())
    sentences_with_triples = sum(1 for data in clean_results.values() if data['triples'])
    
    # 统计不同类型的谓语
    predicate_counts = {}
    for data in clean_results.values():
        for triple in data['triples']:
            predicate = triple[1]
            predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
    
    # 按频率排序谓语
    sorted_predicates = sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)
    
    summary = {
        'total_sentences': total_sentences,
        'total_triples': total_triples,
        'sentences_with_triples': sentences_with_triples,
        'extraction_rate': round(sentences_with_triples / total_sentences * 100, 2) if total_sentences > 0 else 0,
        'avg_triples_per_sentence': round(total_triples / total_sentences, 2) if total_sentences > 0 else 0,
        'top_predicates': sorted_predicates[:10]  # 前10个最频繁的谓语
    }
    
    return summary

def process_triple_extraction(analysis_results):
    """处理三元组提取（包含指代消解）"""
    all_triples = []
    
    for result in analysis_results:
        sentence_num = result['sentence_num']
        analysis = result['analysis']
        
        sentence = analysis['sentence']
        tokens = analysis['tokens']
        pos_tags = analysis['pos_tags']
        dependencies = analysis['dependencies']
        
        print(f"\n{'='*50}")
        print(f"处理句子 {sentence_num}: {sentence}")
        
        # 获取指代消解结果
        coreference_info = get_sentence_coreference_info(sentence)
        coreference_mapping = coreference_info['mapping']
        
        if coreference_mapping:
            print(f"指代消解映射: {coreference_mapping}")
        
        # 获取关键词短语
        keyphrases = get_filtered_keyphrases(sentence)
        print("过滤后的关键词短语:", keyphrases)
        
        # 提取基础三元组
        basic_triples = extract_triples(tokens, pos_tags, dependencies)
        print("基础三元组:", basic_triples)
        
        if basic_triples:
            # 用关键词短语增强三元组
            enhanced_triples = enhance_triples_with_keyphrases_contextual(
                basic_triples, keyphrases, tokens, pos_tags, dependencies)
            print("增强后的三元组:", enhanced_triples)
            
            # 应用指代消解
            if coreference_mapping:
                resolved_triples = resolve_coreference_in_triples(enhanced_triples, coreference_mapping)
                print("指代消解后的三元组:", resolved_triples)
            else:
                resolved_triples = enhanced_triples
            
            # 去重并过滤格式不正确的三元组
            valid_triples = []
            seen_triples = set()  # 使用集合来跟踪已见过的三元组
            
            for triple in resolved_triples:
                if isinstance(triple, tuple) and len(triple) == 3:
                    # 确保三元组的每个元素都是字符串且不为空
                    subject, predicate, obj = triple
                    if all(isinstance(x, str) and x.strip() for x in [subject, predicate, obj]):
                        # 检查是否已经存在
                        if triple not in seen_triples:
                            valid_triples.append(triple)
                            seen_triples.add(triple)
            
            print("去重后的三元组:", valid_triples)
            
            # 按主体在句子中的出现顺序排序
            final_triples = sort_triples_by_subject_order(valid_triples, tokens)
            print("最终三元组:", final_triples)
            
            # 添加到总结果中
            for triple in final_triples:
                all_triples.append((sentence_num, sentence, triple))
        else:
            print("未提取到三元组")
        
        print("=" * 60)
    
    return all_triples

# 主程序
if __name__ == "__main__":
    # 加载句法依存分析结果
    analysis_results = load_dependency_analysis()
    
    if analysis_results is None:
        print("无法加载句法分析结果，程序退出")
        exit(1)
    
    # 提取三元组
    all_triples = process_triple_extraction(analysis_results)
    
    print(f"总共提取了 {len(all_triples)} 个三元组")
    
    # 按句子分组显示结果
    print(f"\n{'='*60}")
    print("按句子分组的结果:")
    print(f"{'='*60}")
    
    # 重新整理数据结构，确保数据清洁
    clean_results = {}
    
    # 清洁和重组数据
    for item in all_triples:
        if len(item) == 3:
            sentence_num, sentence_text, triple = item
            
            # 确保triple是有效的三元组
            if isinstance(triple, tuple) and len(triple) == 3:
                subject, predicate, obj = triple
                
                # 确保所有元素都是字符串
                if all(isinstance(x, str) and x.strip() for x in [subject, predicate, obj]):
                    if sentence_num not in clean_results:
                        clean_results[sentence_num] = {
                            'text': sentence_text,
                            'triples': []
                        }
                    
                    # 检查是否已经存在相同的三元组，避免重复
                    if triple not in clean_results[sentence_num]['triples']:
                        clean_results[sentence_num]['triples'].append(triple)
    
    # 输出清洁后的结果
    for sentence_num in sorted(clean_results.keys()):
        data = clean_results[sentence_num]
        
        print(f"\n句子{sentence_num}: {data['text']}")
        
        if data['triples']:
            print(f"提取的三元组 ({len(data['triples'])} 个):")
            for i, triple in enumerate(data['triples'], 1):
                subject, predicate, obj = triple
                print(f"  {i}. ('{subject}', '{predicate}', '{obj}')")
        else:
            print("提取的三元组 (0 个):")
        
        print("-" * 50)
    
    # 保存三元组结果到JSON文件
    if save_triples_to_json(clean_results):
        print("\n三元组结果保存成功！")
    
    # 生成并显示统计摘要
    summary = generate_summary_statistics(clean_results)
    print(f"\n{'='*60}")
    print("三元组提取统计摘要:")
    print(f"{'='*60}")
    print(f"总句子数: {summary['total_sentences']}")
    print(f"总三元组数: {summary['total_triples']}")
    print(f"有三元组的句子数: {summary['sentences_with_triples']}")
    print(f"提取成功率: {summary['extraction_rate']}%")
    
    print(f"\n{'='*60}")