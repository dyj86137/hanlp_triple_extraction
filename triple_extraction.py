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

def has_digit(s):
    """检查字符串是否包含数字"""
    return any(char.isdigit() for char in s)

def identify_range_expressions(tokens, pos_tags, dependencies):
    """识别句子中的所有范围表达式"""
    range_expressions = []
    range_token_indices = set()  # 追踪已被包含在范围表达式中的token索引
    
    # 第一步：识别连续的数字-分隔符-数字模式
    for i in range(len(tokens) - 2):
        if i in range_token_indices:
            continue
        
        # 检查模式: 数字 + 范围符号 + 数字
        if ((pos_tags[i] == 'CD' or has_digit(tokens[i])) and 
            tokens[i+1] in ['~', '-', '到', '至', '—'] and 
            (pos_tags[i+2] == 'CD' or has_digit(tokens[i+2]))):
            
            range_expr = tokens[i] + tokens[i+1] + tokens[i+2]
            range_expressions.append((range_expr, [i, i+1, i+2]))
            range_token_indices.update([i, i+1, i+2])
            print(f"识别到连续范围表达式: {range_expr}")
    
    # 第二步：识别通过依存关系连接的数值范围
    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        if i in range_token_indices:
            continue
        
        if pos == 'CD' or has_digit(token):
            # 查找范围符号
            for j, (head, relation) in enumerate(dependencies):
                if j >= len(tokens) or j in range_token_indices:
                    continue
                    
                if tokens[j] in ['~', '-', '到', '至', '—'] and (head - 1 == i or j == i + 1):
                    # 查找范围的另一端
                    for k, (head2, relation2) in enumerate(dependencies):
                        if k in range_token_indices or k == i or k == j:
                            continue
                        
                        # 另一个数字可能依赖于范围符号或第一个数字
                        if (pos_tags[k] == 'CD' or has_digit(tokens[k])) and (head2 - 1 == i or head2 - 1 == j or head - 1 == k):
                            # 确保数字-范围符号-数字的合理顺序
                            indices = sorted([i, j, k])
                            range_expr = tokens[indices[0]] + tokens[indices[1]] + tokens[indices[2]]
                            range_expressions.append((range_expr, indices))
                            range_token_indices.update(indices)
                            print(f"识别到依存关系范围表达式: {range_expr}")
                            break
    
    return range_expressions, range_token_indices

def split_complex_sentence(tokens, pos_tags, dependencies):
    """将复合句分割为独立子句"""
    segments = []
    segment_tokens = []
    current_segment = []
    boundaries = []
    
    # 识别句子分隔符（如逗号、分号）和连词
    for i, (token, pos) in enumerate(tokens, pos_tags):
        if token in ['，', '；', '。'] or pos == 'CC':
            boundaries.append(i)
    
    # 使用依存句法树检测子句边界
    for i, (head, relation) in enumerate(dependencies):
        if relation in ['punct', 'conj']:
            if i not in boundaries:
                boundaries.append(i)
    
    # 按边界分割句子
    if not boundaries:
        return [tokens]
        
    boundaries.sort()
    start = 0
    for boundary in boundaries:
        # 确保分段不为空
        if boundary > start:
            segments.append(tokens[start:boundary+1])
        start = boundary + 1
    
    # 添加最后一段
    if start < len(tokens):
        segments.append(tokens[start:])
        
    return segments

def identify_clause_spo(segment_tokens, segment_pos, segment_deps):
    """识别单个子句中的主谓宾结构"""
    subjects = []
    predicates = []
    objects = []
    
    # 识别子句中的谓语动词
    for i, (token, pos) in enumerate(zip(segment_tokens, segment_pos)):
        if pos in ['VV', 'VC']:
            predicates.append((i, token))
            
            # 查找依存于此谓语的主语和宾语
            for j, (head, relation) in enumerate(segment_deps):
                if head - 1 == i:
                    if relation == 'nsubj':
                        subjects.append((j, segment_tokens[j]))
                    elif relation == 'dobj':
                        objects.append((j, segment_tokens[j]))
    
    return subjects, predicates, objects

def has_dependency_path(start_idx, end_idx, dependencies, max_depth=3):
    """检查两个词之间是否存在依存关系路径"""
    # 直接关系检查
    for i, (head, relation) in enumerate(dependencies):
        if i == start_idx and head - 1 == end_idx:
            return True
        if i == end_idx and head - 1 == start_idx:
            return True
    
    # 构建依存关系图
    graph = {}
    for i, (head, relation) in enumerate(dependencies):
        if head == 0:  # 根节点
            continue
        
        from_node = head - 1
        to_node = i
        
        if from_node not in graph:
            graph[from_node] = []
        if to_node not in graph:
            graph[to_node] = []
        
        graph[from_node].append(to_node)
        graph[to_node].append(from_node)  # 双向连接
    
    # 使用广度优先搜索查找路径
    visited = set()
    queue = [(start_idx, 0)]  # (节点, 深度)
    
    while queue:
        node, depth = queue.pop(0)
        
        if node == end_idx:
            return True
        
        if depth >= max_depth:
            continue
        
        if node in visited:
            continue
        
        visited.add(node)
        
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
    
    return False

def check_cross_segment_relation(subject, obj, tokens, dependencies):
    """检查两个不在同一片段的词是否存在有效的跨片段关系"""
    # 找到主语和宾语在原句中的位置
    subj_pos = -1
    obj_pos = -1
    
    for i, token in enumerate(tokens):
        if token == subject or subject in token:
            subj_pos = i
            break
    
    for i, token in enumerate(tokens):
        if token == obj or obj in token:
            obj_pos = i
            break
    
    # 如果找不到位置，无法确认关系
    if subj_pos == -1 or obj_pos == -1:
        return False
    
    # 检查是否有依存路径
    return has_dependency_path(subj_pos, obj_pos, dependencies)

def identify_topic_information(tokens, pos_tags, dependencies):
    """
    识别句子中的话题/背景信息（如"对于X"结构）
    需要将话题/背景信息也加入三元组中
    """
    topic_info = None
    topic_scope = "global"  # 默认话题应用于整个句子
    
    # 简单模式匹配：如果句子以"对于"开头，提取到第一个逗号之间的内容
    if tokens and tokens[0] in ['对于', '关于', '就']:
        for i, token in enumerate(tokens):
            if token == "，":
                topic_info = "".join(tokens[1:i])
                print(f"通过模式匹配识别到话题: {topic_info}")
                return topic_info, topic_scope
    
    # 依存关系方法（备用）
    # 寻找由介词"对于"/"关于"等引导的话题
    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        if token in ['对于', '关于', '就'] and pos == 'P':
            # 打印依存关系调试信息
            print(f"检查话题介词 '{token}' 在索引 {i} 的依存关系")
            
            # 收集从介词到第一个逗号的所有名词作为话题短语
            j = i + 1
            topic_words = []
            while j < len(tokens) and tokens[j] != "，":
                if pos_tags[j] in ['NN', 'NR', 'JJ', 'CD']:
                    topic_words.append(tokens[j])
                j += 1
            
            if topic_words:
                topic_info = "".join(topic_words)
                print(f"通过依存关系识别到话题: {topic_info}")
                return topic_info, topic_scope
    
    return topic_info, topic_scope

def build_compound_subject(direct_subject, topic_info, tokens, pos_tags, dependencies):
    """构建包含话题/背景信息的复合主语"""
    if not topic_info:
        return direct_subject
        
    # 查找主语和话题在原文中的位置
    subj_pos = -1
    for i, token in enumerate(tokens):
        if token == direct_subject or direct_subject in token:
            subj_pos = i
            break
    
    # 检查直接主语是否已经包含话题信息
    if topic_info in direct_subject:
        return direct_subject
    
    # 检查主语和话题之间是否存在依存关系
    has_relation = False
    for i, (head, rel) in enumerate(dependencies):
        if topic_info in tokens[i] and has_dependency_path(i, subj_pos, dependencies, 3):
            has_relation = True
            break
    
    # 根据不同条件构建复合主语
    if has_relation:
        # 如果有明确的依存关系，使用"的"连接
        return topic_info + "的" + direct_subject
    else:
        # 如果是全局话题，也使用"的"连接
        return topic_info + "的" + direct_subject

def normalize_compound_subject(subject):
    """规范化复合主语，避免重复信息"""
    # 处理可能的重复片段
    parts = subject.split('的')
    if len(parts) >= 2:
        # 检查后一部分是否已经包含前一部分信息
        if parts[0] in parts[1]:
            return parts[1]
    return subject

def enforce_semantic_boundary(triples, sentence, tokens, dependencies):
    """确保三元组的语义边界正确"""
    validated_triples = []
    
    # 按逗号分割句子获取大致的语义片段
    segments = sentence.split('，')
    
    for subject, predicate, obj in triples:
        # 检查主语和客体是否在同一语义片段中
        found_in_same_segment = False
        for segment in segments:
            if subject in segment and obj in segment:
                found_in_same_segment = True
                break
        
        # 如果两者在同一片段或有明确的跨片段关系，保留三元组
        if found_in_same_segment or check_cross_segment_relation(subject, obj, tokens, dependencies):
            validated_triples.append((subject, predicate, obj))
        else:
            print(f"过滤语义范围不匹配的三元组: ({subject}, {predicate}, {obj})")
    
    return validated_triples

def anchor_subjects_to_predicates(tokens, pos_tags, dependencies):
    """将主语锚定到正确的谓语上"""
    subject_predicate_map = {}
    
    # 识别句子中的关键标点分隔符
    separators = [i for i, token in enumerate(tokens) if token in ['，', '；', '。']]
    segments = []
    
    # 根据分隔符划分句子段落
    if separators:
        start = 0
        for sep in separators:
            segments.append((start, sep))
            start = sep + 1
        segments.append((start, len(tokens)))
    else:
        segments = [(0, len(tokens))]
    
    # 针对每个段落识别主语-谓语对应关系
    for start, end in segments:
        segment_subjects = []
        segment_predicates = []
        
        # 找出段落内的主语和谓语
        for i in range(start, end):
            if pos_tags[i] in ['VV', 'VC']:
                segment_predicates.append(i)
                
            for j, (head, relation) in enumerate(dependencies):
                if start <= j < end and head - 1 == i and relation == 'nsubj':
                    segment_subjects.append((j, i))  # (主语索引，谓语索引)
        
        # 在该段落内，将主语与对应谓语配对
        for subj_idx, pred_idx in segment_subjects:
            subject_predicate_map[subj_idx] = pred_idx
    
    return subject_predicate_map

def validate_triple_with_context(subject, predicate, obj, tokens, pos_tags, dependencies):
    """根据上下文验证三元组的合理性"""
    # 查找主语、谓语和宾语在原文中的位置
    subj_pos = -1
    pred_pos = -1
    obj_pos = -1
    
    for i, token in enumerate(tokens):
        if token == subject or subject in token:
            subj_pos = i
        if token == predicate or predicate in token:
            pred_pos = i
        if token == obj or obj in token:
            obj_pos = i
    
    # 如果主语/宾语在原文中能找到位置
    if subj_pos >= 0 and obj_pos >= 0:
        # 检查主语和宾语是否在同一逗号分隔的片段内
        separators = [i for i, t in enumerate(tokens) if t in ['，', '；', '。']]
        
        # 找到主语和宾语所在的片段
        subj_segment = next((i for i, sep in enumerate(separators) if subj_pos < sep), len(separators))
        obj_segment = next((i for i, sep in enumerate(separators) if obj_pos < sep), len(separators))
        
        # 如果它们在不同片段，检查是否可能存在正确的关系
        if subj_segment != obj_segment:
            # 检查依存关系是否支持这个三元组
            if not has_dependency_path(subj_pos, obj_pos, dependencies):
                print(f"主语和宾语在不同片段且无依存关系: {subject}, {obj}")
                return False
    
    return True

def extract_triples_from_complex_sentence(sentence, tokens, pos_tags, dependencies):
    """从复合句中提取三元组，处理多子句情况"""
    all_triples = []
    
    # 1. 首先尝试整体提取
    basic_triples = extract_triples(tokens, pos_tags, dependencies)
    
    # 2. 分割复合句
    segments = split_complex_sentence(tokens, pos_tags, dependencies)
    
    # 仅在有多个子句时进行子句处理
    if len(segments) > 1:
        # 3. 锚定主语与谓语关系
        subject_predicate_map = anchor_subjects_to_predicates(tokens, pos_tags, dependencies)
        
        # 4. 对每个子句单独处理
        segment_triples = []
        for segment in segments:
            # 提取该子句的tokens, pos和dependencies
            seg_start = segment[0]
            seg_end = segment[-1]
            segment_tokens = tokens[seg_start:seg_end+1]
            segment_pos = pos_tags[seg_start:seg_end+1]
            
            # 调整依存关系索引
            segment_deps = []
            for i, (head, rel) in enumerate(dependencies):
                if seg_start <= i <= seg_end:
                    new_head = head - seg_start if head > 0 else head
                    segment_deps.append((new_head, rel))
                    
            # 提取子句三元组
            sub_triples = extract_triples(segment_tokens, segment_pos, segment_deps)
            if sub_triples:
                segment_triples.extend(sub_triples)
        
        # 5. 合并结果并验证
        all_candidates = basic_triples + segment_triples
        
        # 6. 对所有候选三元组进行上下文验证
        for triple in all_candidates:
            subject, predicate, obj = triple
            if validate_triple_with_context(subject, predicate, obj, tokens, pos_tags, dependencies):
                all_triples.append(triple)
    else:
        # 句子没有多个子句，使用基础提取结果
        all_triples = basic_triples
    
    # 7. 进行最终的语义范围约束
    # 7. 进行最终的语义范围约束
    validated_triples = enforce_semantic_boundary(all_triples, sentence, tokens, dependencies)
    
    return list(set(validated_triples))  # 去重

def extract_complete_phrase(start_idx, tokens, dependencies, pos_tags):
    """收集所有修饰给定词的形容词、限定词和复合词"""
    if start_idx < 0 or start_idx >= len(tokens):
        return ""
        
    phrase_components = [(start_idx, tokens[start_idx])]  # 包含目标词本身
    visited = set([start_idx])
    
    # 1. 递归收集所有依赖于当前节点的修饰词
    def collect_modifiers(node_idx, depth=0, max_depth=3):
        if depth > max_depth or node_idx in visited:
            return []
        
        modifiers = []
        for i, (head, relation) in enumerate(dependencies):
            if i >= len(tokens):
                continue
                
            if head - 1 == node_idx and i not in visited:
                # 包含更多修饰关系
                if relation in ['amod', 'advmod', 'det', 'compound:nn', 'nmod:assmod', 'dep', 'mark']:
                    modifiers.append((i, tokens[i]))
                    visited.add(i)
                    # 递归收集修饰词的修饰词
                    modifiers.extend(collect_modifiers(i, depth+1))
                    
                # 特别处理"的"字结构
                elif relation == 'case' and tokens[i] == '的':
                    modifiers.append((i, tokens[i]))
                    visited.add(i)
        
        return modifiers
    
    # 收集目标词的所有修饰词
    phrase_components.extend(collect_modifiers(start_idx))
    
    # 2. 反向处理: 如果当前词是修饰词，也获取它修饰的词
    for i, (head, relation) in enumerate(dependencies):
        if i == start_idx and relation in ['amod', 'advmod', 'compound:nn', 'mark']:
            target_idx = head - 1
            if target_idx >= 0 and target_idx < len(tokens) and target_idx not in visited:
                phrase_components.append((target_idx, tokens[target_idx]))
                visited.add(target_idx)
                # 继续收集这个被修饰词的其他修饰词
                phrase_components.extend(collect_modifiers(target_idx))
    
    # 3. 特殊处理连续的形容词-"的"-名词结构
    for i, token in enumerate(tokens):
        if token == "的" and i > 0 and i < len(tokens) - 1:
            # 检查"的"前面是否是形容词，后面是否是名词
            if i-1 in visited or i+1 in visited:  # 如果"的"的前后词有一个在已收集集合中
                if i-1 not in visited:
                    phrase_components.append((i-1, tokens[i-1]))
                    visited.add(i-1)
                if i not in visited:
                    phrase_components.append((i, tokens[i]))
                    visited.add(i)
                if i+1 not in visited:
                    phrase_components.append((i+1, tokens[i+1]))
                    visited.add(i+1)
    
    # 4. 特殊处理形容词前的副词(如"最常用的")
    for idx, token in phrase_components:
        if idx > 0:
            prev_idx = idx - 1
            if prev_idx not in visited and prev_idx < len(tokens):
                prev_pos = pos_tags[prev_idx]
                if prev_pos == 'AD':  # 副词
                    phrase_components.append((prev_idx, tokens[prev_idx]))
                    visited.add(prev_idx)
    
    # 按原文顺序排列并去重
    unique_components = []
    for idx, token in phrase_components:
        if 0 <= idx < len(tokens):  # 确保索引有效
            unique_components.append((idx, token))
    
    # 排序并构建短语
    unique_components.sort()
    phrase = ''.join([token for _, token in unique_components])
    return phrase

def get_complete_prep_phrase(prep_idx, tokens, pos_tags, dependencies):
    """提取任意介词引导的完整短语"""
    # 首先获取介词
    if prep_idx >= len(tokens):
        return ""
    
    prep = tokens[prep_idx]
    
    # 收集介词短语的所有成分
    phrase_components = [(prep_idx, prep)]
    
    # 1. 收集直接依赖于介词的成分
    direct_deps = []
    for i, (head, relation) in enumerate(dependencies):
        if head - 1 == prep_idx:
            direct_deps.append((i, tokens[i], relation))
    
    # 2. 递归收集所有依赖成分
    def collect_dependents(node_idx):
        result = []
        for i, (head, relation) in enumerate(dependencies):
            if head - 1 == node_idx and i != prep_idx:
                result.append((i, tokens[i]))
                # 递归收集依赖项的依赖
                result.extend(collect_dependents(i))
        return result
    
    # 为直接依赖项收集所有传递性依赖项
    for dep_idx, _, _ in direct_deps:
        phrase_components.extend(collect_dependents(dep_idx))
        phrase_components.append((dep_idx, tokens[dep_idx]))
    
    # 3. 查找介词短语的边界（如"之间"这类方位词）
    for i, (head, relation) in enumerate(dependencies):
        if relation == 'case' and i > prep_idx:
            scope_end = i
            for j, (head2, rel2) in enumerate(dependencies):
                if head2 - 1 == scope_end:
                    phrase_components.append((j, tokens[j]))
    
    # 按原文顺序排列并移除重复项
    seen = set()
    unique_components = []
    for idx, token in phrase_components:
        if idx not in seen:
            seen.add(idx)
            unique_components.append((idx, token))
    
    unique_components.sort()
    
    # 构建短语文本
    phrase = ''.join([token for _, token in unique_components])
    return phrase

def extract_triples(tokens, pos_tags, dependencies):
    """基于依存句法提取三元组，增强对范围表达式、从属关系、修饰词的识别"""
    triples = []
    
    # 首先识别所有范围表达式
    range_expressions, range_token_indices = identify_range_expressions(tokens, pos_tags, dependencies)
    
    if range_expressions:
        print(f"识别到的范围表达式: {[expr for expr, _ in range_expressions]}")

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
                        # 查找并添加可能的修饰词
                        full_subject = extract_complete_phrase(j, tokens, dependencies, pos_tags)
                        verb_subjects.append((j, full_subject))
                    elif relation == 'dobj':
                        verb_objects.append((j, tokens[j]))
                    # 防止将nmod:prep关系的数值并入谓语
                    elif relation == 'nmod:prep' and pos_tags[j] == 'CD':
                        # 不要在此处处理，保留到后续单独处理介词短语
                        continue
            
            # 如果是系动词(VC)，特殊处理
            if pos == 'VC':
                # 针对主谓宾结构，寻找真正的主语（考虑"的"结构）
                real_subject_pairs = []
                for subj_idx, subject in verb_subjects:
                    # 查找是否有名词通过"的"修饰当前主语
                    true_subject = None
                    for k, (head, relation) in enumerate(dependencies):
                        # 如果某词依赖于当前主语，且关系是所有格(nmod:assmod)
                        if head - 1 == subj_idx and relation == 'nmod:assmod':
                            # 再检查是否有"的"连接它们
                            for l, (head2, rel2) in enumerate(dependencies):
                                if head2 - 1 == k and rel2 == 'case' and tokens[l] == '的':
                                    # 找到真实主语
                                    true_subj_text = tokens[k]
                                    
                                    # 如果真实主语前面有修饰词（如"直径"），尝试组合它们
                                    for m, (head3, rel3) in enumerate(dependencies):
                                        if head3 - 1 == k and rel3 in ['nsubj', 'compound:nn']:
                                            true_subj_text = tokens[m] + true_subj_text
                                    
                                    # 记录原主语和真实主语
                                    print(f"发现真实主语: {true_subj_text} 取代 {subject}")
                                    true_subject = true_subj_text
                                    real_subject_pairs.append((true_subject, subject))
                                    break
                            break
                    
                    if true_subject is None:
                        # 如果没找到真正的主语，使用原始主语
                        real_subject_pairs.append((subject, None))
                
                # 处理范围表达式
                for range_expr, indices_list in range_expressions:
                    # 检查范围表达式是否与当前系动词相关联
                    is_related = False
                    
                    for idx in indices_list:
                        for j, (head, relation) in enumerate(dependencies):
                            if j == idx and (head - 1 == i or relation in ['dep', 'punct']):
                                is_related = True
                                break
                        if is_related:
                            break
                    
                    # 还需检查是否有数字直接依赖于系动词
                    if not is_related:
                        for idx in indices_list:
                            if pos_tags[idx] == 'CD':
                                for j, (head, relation) in enumerate(dependencies):
                                    if j == idx and head - 1 == i:
                                        is_related = True
                                        break
                            if is_related:
                                break
                    
                    # 如果范围表达式与系动词相关，形成三元组
                    # if is_related and real_subject_pairs:
                    #     for real_subject, orig_subject in real_subject_pairs:
                    #         if orig_subject is not None:
                    #             # 使用"真实主语"作为主语，将原主语加入谓语
                    #             triple = (real_subject, orig_subject + verb, range_expr)
                    #         else:
                    #             # 使用原始主语
                    #             triple = (real_subject, verb, range_expr)
                            
                    #         if triple not in triples:
                    #             triples.append(triple)
                    #             print(f"添加属性范围三元组: {triple}")
                
                # 常规系动词处理
                cop_head_idx = -1
                for j, (head, relation) in enumerate(dependencies):
                    if j == i and relation == 'cop':
                        cop_head_idx = head - 1  # 系动词指向的宾语
                        break
                
                if cop_head_idx >= 0:
                    # 找这个宾语的主语
                    for j, (head, relation) in enumerate(dependencies):
                        if head - 1 == cop_head_idx and relation == 'nsubj':
                            subject = extract_complete_phrase(j, tokens, dependencies, pos_tags)  # 提取完整主语
                            predicate_obj = extract_complete_phrase(cop_head_idx, tokens, dependencies, pos_tags)  # 提取完整宾语
                            triples.append((subject, verb, predicate_obj))
            
            # 处理一般动词(VV)
            elif pos == 'VV':
                # 处理动词关联的介词短语 - 保留这个正确的处理逻辑
                for j, (head, relation) in enumerate(dependencies):
                    if head - 1 == i and relation == 'nmod:prep':
                        # 获取介词
                        prep_idx = j

                        # 查找与此动词相关的主语
                        subject_found = False
                        for m, (head2, rel2) in enumerate(dependencies):
                            if head2 - 1 == i and rel2 == 'nsubj':
                                subject = tokens[m]
                                # 只使用动词作为谓语
                                predicate = verb
                                # 使用完整的介词短语作为宾语
                                prep_phrase = get_complete_prep_phrase(prep_idx, tokens, pos_tags, dependencies)
                                triple = (subject, predicate, prep_phrase)
                                if triple not in triples:
                                    triples.append(triple)
                                    print(f"添加介词短语三元组: {triple}")
                                subject_found = True

                        # 如果没有找到主语，可能需要从更广的范围查找
                        if not subject_found:
                            # 查找句子中任何可能的主语
                            for m, (head2, rel2) in enumerate(dependencies):
                                if rel2 == 'nsubj':
                                    # 检查这个主语是否与当前动词在合理的依存路径上
                                    subject = tokens[m]
                                    predicate = verb
                                    prep_phrase = get_complete_prep_phrase(prep_idx, tokens, pos_tags, dependencies)
                                    triple = (subject, predicate, prep_phrase)
                                    if triple not in triples:
                                        triples.append(triple)
                                        print(f"添加推断主语介词短语三元组: {triple}")
                                    break

                # 处理范围表达式
                for range_expr, indices_list in range_expressions:
                    # 检查范围表达式是否作为宾语依存于当前动词
                    for idx in indices_list:
                        for j, (head, relation) in enumerate(dependencies):
                            if j == idx and head - 1 == i and relation in ['dobj', 'dep']:
                                if verb_subjects:  # 如果有明确的主语
                                    for _, subject in verb_subjects:
                                        triples.append((subject, verb, range_expr))
                                break
                
                # 原有的逻辑处理
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
                        for _, obj in verb_objects:
                            triples.append((subj, verb, obj))
                
                # 如果有明确的主语和宾语
                elif verb_subjects and verb_objects:
                    for _, subj in verb_subjects:
                        for _, obj in verb_objects:
                            triples.append((subj, verb, obj))
    
    # 再次检查范围表达式，确保与系动词正确关联
    if 'VC' in pos_tags and range_expressions:
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            if pos == 'VC':
                # 找到与此系动词相关的主语
                subjects_for_this_verb = []
                real_subject_pairs = []
                
                for j, (head, relation) in enumerate(dependencies):
                    if head - 1 == i and relation == 'nsubj':
                        # 查找是否有名词通过"的"修饰当前主语
                        has_real_subject = False
                        for k, (head2, relation2) in enumerate(dependencies):
                            if head2 - 1 == j and relation2 == 'nmod:assmod':  # 找到依存于主语的定语
                                # 再检查是否有"的"连接它们
                                for l, (head3, rel3) in enumerate(dependencies):
                                    if head3 - 1 == k and rel3 == 'case' and tokens[l] == '的':
                                        # 构建真实主语短语
                                        true_subject = tokens[k]
                                        
                                        # 检查是否有其他词修饰真实主语
                                        subject_indices = [k]
                                        for m, (head4, rel4) in enumerate(dependencies):
                                            if head4 - 1 == k and m != l:
                                                subject_indices.append(m)
                                        
                                        # 按原始顺序排列
                                        subject_components = [(idx, tokens[idx]) for idx in subject_indices]
                                        subject_components.sort()
                                        true_subject = ''.join(token for _, token in subject_components)
                                        
                                        real_subject_pairs.append((true_subject, tokens[j]))
                                        has_real_subject = True
                                        break
                                if has_real_subject:
                                    break
                        
                        if not has_real_subject:
                            subjects_for_this_verb.append(tokens[j])
                
                # 查找可能的范围表达式
                for range_expr, indices in range_expressions:
                    # 检查是否已经为此范围表达式创建了三元组
                    is_processed = False
                    for real_subj, orig_subj in real_subject_pairs:
                        if any((real_subj, orig_subj + token, range_expr) == triple for triple in triples):
                            is_processed = True
                            break
                    
                    if is_processed:
                        continue
                        
                    # 检查范围表达式中的数字是否与此系动词相关联
                    for idx in indices:
                        if pos_tags[idx] == 'CD':
                            # 检查是否有数字依赖于此系动词
                            dep_relation = False
                            for j, (head, relation) in enumerate(dependencies):
                                if j == idx and head - 1 == i:
                                    dep_relation = True
                                    break
                            
                            if dep_relation:
                                # 优先使用真实主语
                                if real_subject_pairs:
                                    for real_subj, orig_subj in real_subject_pairs:
                                        triple = (real_subj, orig_subj + token, range_expr)
                                        if triple not in triples:
                                            triples.append(triple)
                                            print(f"添加真实主语范围三元组: {triple}")
                                elif subjects_for_this_verb:
                                    for subj in subjects_for_this_verb:
                                        triple = (subj, token, range_expr)
                                        if triple not in triples:
                                            triples.append(triple)
    
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

def find_best_keyphrase_match_with_context(word, word_index, keyphrases, tokens, predicate=None, object_value=None):
    """根据上下文位置找到最佳匹配的关键词短语，考虑谓语和宾语信息"""
    print(f"  匹配词汇: {word} (索引: {word_index})")
    
    # 1. 精确匹配
    if word in keyphrases:
        print(f"  精确匹配: {word}")
        return word
    
    # 1.5 如果有宾语信息，先检查同一语义段落
    if object_value:
        segments = []
        current_seg = []
        
        # 按逗号分段
        for i, token in enumerate(tokens):
            current_seg.append(token)
            if token in ['，', '；', '。']:
                segments.append(''.join(current_seg))
                current_seg = []
        
        if current_seg:
            segments.append(''.join(current_seg))
        
        # 查看哪些候选词和宾语在同一语义段落
        segment_candidates = []
        for kp in keyphrases:
            if word in kp:
                for segment in segments:
                    if kp in segment and object_value in segment:
                        segment_candidates.append(kp)
                        print(f"  语义段落匹配: {kp} 与 {object_value} 同段")
                        
        if segment_candidates:
            return max(segment_candidates, key=len)  # 返回最长的匹配
    
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

def process_triple_extraction(analysis_results):
    """处理三元组提取（包含指代消解和话题信息）"""
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
        
        # 1. 先识别话题信息
        topic_info, topic_scope = identify_topic_information(tokens, pos_tags, dependencies)
        print(f"识别到的话题信息: {topic_info}, 范围: {topic_scope}")
        
        # 2. 提取基础三元组
        basic_triples = extract_triples(tokens, pos_tags, dependencies)
        print("基础三元组:", basic_triples)
        
        if basic_triples:
            # 3. 用关键词短语增强三元组
            enhanced_triples = enhance_triples_with_keyphrases_contextual(
                basic_triples, keyphrases, tokens, pos_tags, dependencies)
            print("增强后的三元组:", enhanced_triples)
            
            # 4. 添加话题信息
            if topic_info:
                print(f"准备添加话题信息: {topic_info}")
                topic_enhanced_triples = []
                for subject, predicate, obj in enhanced_triples:
                    # 构建复合主语 - 避免重复信息
                    if topic_info not in subject:
                        compound_subject = topic_info + "的" + subject
                        compound_subject = normalize_compound_subject(compound_subject)
                        print(f"添加话题到主语: {subject} -> {compound_subject}")
                        topic_enhanced_triples.append((compound_subject, predicate, obj))
                    else:
                        topic_enhanced_triples.append((subject, predicate, obj))
                enhanced_triples = topic_enhanced_triples
                print(f"添加话题信息后的三元组: {enhanced_triples}")
            
            # 5. 应用指代消解
            if coreference_mapping:
                resolved_triples = resolve_coreference_in_triples(enhanced_triples, coreference_mapping)
                print("指代消解后的三元组:", resolved_triples)
            else:
                resolved_triples = enhanced_triples
            
            # 6. 去重并过滤格式不正确的三元组
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
            
            # 7. 按主体在句子中的出现顺序排序
            final_triples = sort_triples_by_subject_order(valid_triples, tokens)
            print("最终三元组:", final_triples)
            
            # 8. 添加到总结果中
            for triple in final_triples:
                all_triples.append((sentence_num, sentence, triple))
        else:
            print("未提取到三元组")
        
        print("=" * 60)
    
    return all_triples

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
        
        # 根据位置找到最佳匹配，传递宾语信息以提供上下文
        enhanced_subject = find_best_keyphrase_match_with_context(
            subject, subject_index, keyphrases, tokens, predicate, obj)
        enhanced_object = find_best_keyphrase_match_with_context(
            obj, object_index, keyphrases, tokens, predicate)
        
        enhanced_triples.append((enhanced_subject, predicate, enhanced_object))
        print(f"增强结果: ({enhanced_subject}, {predicate}, {enhanced_object})")
    
    return enhanced_triples

def validate_triples_semantically(triples, sentence):
    """验证三元组的语义合理性，解决矛盾的关系"""
    validated_triples = []
    seen_relationships = {}  # 存储已经看到的主语-谓语对
    
    # 按逗号分割句子获取大致的语义片段
    segments = sentence.split('，')
    
    for triple in triples:
        subject, predicate, obj = triple
        
        # 检查主语和客体是否在同一语义片段中
        in_same_segment = False
        for segment in segments:
            if subject in segment and obj in segment:
                in_same_segment = True
                break
        
        # 如果在同一片段中，则优先保留
        if in_same_segment:
            key = (subject, predicate)
            seen_relationships[key] = triple
        else:
            # 如果不在同一片段，看看是否已经有同一主语-谓语对
            key = (subject, predicate)
            if key not in seen_relationships:
                seen_relationships[key] = triple
    
    return list(seen_relationships.values())

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