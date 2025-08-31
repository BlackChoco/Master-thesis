import torch
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union # Added Union
from Tree_data_model import PostStorage, ForestManager
import pickle
from collections import defaultdict, Counter # Added Counter
from tqdm import tqdm
import copy
# 修改导入：使用ModelScope替代transformers
from modelscope import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig
import pandas as pd
import json
import itertools
import jieba # Added jieba
import os # Added os
import torch.nn.functional as F # Added F for TextCNN
import matplotlib.pyplot as plt # 新增
import io # 新增
import matplotlib.image as mpimg # 新增
import warnings
import re # 新增

# 此设置将抑制 (ignore) 消息内容匹配特定模式的 UserWarning 类型的警告
# 具体来说，它针对的是 matplotlib 库中关于 Arial 字体缺少字形 (Glyph) 而产生的警告
warnings.filterwarnings("ignore", category=UserWarning, message=r"Glyph .* missing from font\(s\) Arial\.")

def preprocess_text(text: str) -> str:
    """
    对文本进行预处理：
    1. 去除方括号内的表情符号，如 '[笑哭]'。
    2. 去除指定的品牌名称。
    3. 返回处理后并去除首尾空格的文本。
    """
    if not isinstance(text, str):
        return ""
    
    # 1. 正则匹配'[]'去掉表情符号
    processed_text = re.sub(r'\[.*?\]', '', text)

    # --- 新增：去掉@用户名的文本 ---
    # 这个正则表达式会匹配 '@' 符号，后面跟着一个或多个非空格字符，
    # 最后可能跟着一个空格。
    processed_text = re.sub(r'@\S+\s?', '', processed_text)

    # 2. 去掉厂商的名字
    brand_names = [
        '小米', '苹果', '三星', '荣耀', '华为', '一加', 
        'oppo', 'OPPO', 'vivo', 'realme', '红米', '真我','安卓',
        'x7pro','x100s','gt5pro','GTneo5','pro','手机','11','12','13','14','备用机',
        'p40'
    ]
    # 使用正则表达式一次性替换所有品牌名（不区分大小写）
    brand_pattern = re.compile('|'.join(brand_names), re.IGNORECASE)
    processed_text = brand_pattern.sub('', processed_text)
    
    return processed_text.strip()

# 添加ModelScope辅助函数
def load_model_from_modelscope(model_name_or_path: str, trust_remote_code: bool = True, **kwargs):
    """
    优先从ModelScope加载模型，失败时提示用户
    """
    try:
        print(f"🔍 正在从ModelScope加载模型: {model_name_or_path}")
        model = AutoModel.from_pretrained(
            model_name_or_path, 
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        print(f"✅ 成功从ModelScope加载模型: {model_name_or_path}")
        return model
    except Exception as e:
        print(f"❌ 从ModelScope加载模型失败: {model_name_or_path}")
        print(f"错误信息: {e}")
        print("请检查模型名称是否正确，或网络连接是否正常。")
        raise e

def load_tokenizer_from_modelscope(model_name_or_path: str, trust_remote_code: bool = True, **kwargs):
    """
    优先从ModelScope加载分词器，失败时提示用户
    """
    try:
        print(f"🔍 正在从ModelScope加载分词器: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        print(f"✅ 成功从ModelScope加载分词器: {model_name_or_path}")
        return tokenizer
    except Exception as e:
        print(f"❌ 从ModelScope加载分词器失败: {model_name_or_path}")
        print(f"错误信息: {e}")
        print("请检查模型名称是否正确，或网络连接是否正常。")
        raise e
    
# Helper function to build vocabulary
def build_vocab_from_post_storage(post_storage: PostStorage, min_freq: int = 1) -> Tuple[Dict[str, int], int]:
    """
    使用jieba从PostStorage中的所有评论内容构建词汇表。
    """
    print("正在使用 jieba 构建词汇表...")
    word_counts = Counter()
    
    all_comments_content = []
    for post_id, post_tree in post_storage.posts.items():
        def collect_content_from_node(node):
            if node.content and isinstance(node.content, str): # 确保内容是字符串
                all_comments_content.append(node.content)
            else:
                # 如果内容不是字符串或为空，可以记录或跳过
                # print(f"警告: 节点 {node.comment_id} 在帖子 {post_id} 中的内容不是有效字符串: {node.content}")
                pass # 或者 all_comments_content.append("") 如果希望空字符串参与
            for child in node.children:
                collect_content_from_node(child)
        if post_tree.root: # 确保根节点存在
            collect_content_from_node(post_tree.root)

    print(f"找到 {len(all_comments_content)} 条评论用于构建词汇表。")
    
    for text in tqdm(all_comments_content, desc="评论分词中"):
        if text and isinstance(text, str): # 再次确保文本是有效字符串
            try:
                seg_list = jieba.lcut(text.strip())
                word_counts.update(seg_list)
            except Exception as e:
                print(f"警告: jieba 分词失败，文本: '{text[:50]}...'，错误: {e}")
        elif not text:
            pass # 跳过空文本
        else:
            print(f"警告: 无效的文本类型进行分词: {type(text)}, 内容: {text[:50]}...")


    # 创建词汇表
    # 特殊标记: <pad> 用于填充, <unk> 用于未知词
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2 # 从2开始索引
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    print(f"词汇表构建完成，包含 {len(vocab)} 个独立词元 (min_freq={min_freq})。")
    return vocab, len(vocab)

class TextCNNTokenizer:
    """
    使用预构建词汇表和jieba的TextCNN模型分词器。
    """
    def __init__(self, word_to_idx: Dict[str, int], max_length: int = 128):
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_token_id = word_to_idx.get(self.pad_token, 0) # 提供默认值以防万一
        self.unk_token_id = word_to_idx.get(self.unk_token, 1) # 提供默认值

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            text = str(text) # 尝试转换为字符串
        return jieba.lcut(text.strip())

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.word_to_idx.get(token, self.unk_token_id) for token in tokens]

    def _pad_truncate_single(self, token_ids: List[int], max_len: int) -> Tuple[List[int], List[int]]:
        attention_mask = [1] * len(token_ids)
        if len(token_ids) < max_len:
            padding_len = max_len - len(token_ids)
            token_ids.extend([self.pad_token_id] * padding_len)
            attention_mask.extend([0] * padding_len)
        elif len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            attention_mask = attention_mask[:max_len]
        return token_ids, attention_mask

    def __call__(self, texts: Union[str, List[str]], padding: Union[bool, str] = True, 
                 truncation: Union[bool, str] = True, return_tensors: Optional[str] = None, 
                 max_length: Optional[int] = None) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        if isinstance(texts, str):
            texts = [texts]
        
        # 确保所有输入都是字符串
        processed_texts = []
        for i, text_input in enumerate(texts):
            if not isinstance(text_input, str):
                # print(f"警告: 输入文本 {i} 不是字符串 (类型: {type(text_input)}), 将尝试转换为字符串。内容: {str(text_input)[:50]}...")
                processed_texts.append(str(text_input))
            else:
                processed_texts.append(text_input)
        texts = processed_texts
        
        effective_max_length = max_length if max_length is not None else self.max_length

        batch_input_ids = []
        batch_attention_masks = []

        for text in texts:
            tokens = self.tokenize(text)
            token_ids = self.convert_tokens_to_ids(tokens)
            
            # 根据 truncation 和 padding 参数处理
            if truncation:
                token_ids = token_ids[:effective_max_length]
            
            # 只有在 padding 为 True 时才进行填充
            if padding:
                # 如果 truncation=True, token_ids 已经被截断到 effective_max_length
                # 如果 truncation=False, token_ids 可能仍然超过 effective_max_length, 但我们只填充到 effective_max_length
                current_len = len(token_ids)
                attention_mask = [1] * current_len
                if current_len < effective_max_length:
                    pad_len = effective_max_length - current_len
                    token_ids.extend([self.pad_token_id] * pad_len)
                    attention_mask.extend([0] * pad_len)
                elif current_len > effective_max_length: # 理论上如果 truncation=True 不会发生
                    token_ids = token_ids[:effective_max_length]
                    attention_mask = [1] * effective_max_length

                padded_ids = token_ids
            else: #不填充
                padded_ids = token_ids 
                attention_mask = [1] * len(token_ids)


            batch_input_ids.append(padded_ids)
            batch_attention_masks.append(attention_mask)

        if return_tensors == 'pt':
            # 如果 padding=False，且序列长度不一致，torch.tensor 会报错
            # 在这种情况下，HuggingFace 通常会要求用户使用 DataCollator
            # 这里，如果 padding=False 但要求 pt，我们填充到批次中的最大长度
            if not padding:
                max_len_in_batch = 0
                if batch_input_ids: # 确保列表不为空
                    max_len_in_batch = max(len(ids) for ids in batch_input_ids) if batch_input_ids else 0
                
                if max_len_in_batch > 0 : # 仅当有实际数据时操作
                    for i in range(len(batch_input_ids)):
                        ids = batch_input_ids[i]
                        mask = batch_attention_masks[i]
                        len_diff = max_len_in_batch - len(ids)
                        if len_diff > 0:
                            batch_input_ids[i] = ids + [self.pad_token_id] * len_diff
                            batch_attention_masks[i] = mask + [0] * len_diff
            try:
                batch_input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
                batch_attention_masks_tensor = torch.tensor(batch_attention_masks, dtype=torch.long)
            except RuntimeError as e:
                print(f"将列表转换为张量时出错。检查序列长度是否一致。错误: {e}")
                print(f"批处理输入ID长度: {[len(ids) for ids in batch_input_ids]}")
                # 可以选择抛出错误或尝试进一步处理/调试
                raise e

            return {
                'input_ids': batch_input_ids_tensor,
                'attention_mask': batch_attention_masks_tensor
            }
        else: # 返回列表
            return {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_masks
            }

class TextCNNModel(torch.nn.Module):
    """
    用于句子嵌入的TextCNN模型。
    """
    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int, 
                 filter_sizes: List[int], output_dim_for_encoder: int, dropout_rate: float = 0.1):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=embedding_dim, 
                            out_channels=num_filters, 
                            kernel_size=K) 
            for K in filter_sizes
        ])
        self.dropout = torch.nn.Dropout(dropout_rate)
        # TextCNN本身的输出维度，在ContrastiveEncoder中的投影头之前
        self.fc = torch.nn.Linear(len(filter_sizes) * num_filters, output_dim_for_encoder)
        self.base_dim = output_dim_for_encoder # <--- 添加这一行

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # (可选) 应用 attention_mask 到嵌入层，将填充位置的嵌入置零
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1).float()

        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved[i]: [batch_size, num_filters, seq_len - filter_sizes[i] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled[i]: [batch_size, num_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1)) # [batch_size, num_filters * len(filter_sizes)]
        output = self.fc(cat) # [batch_size, output_dim_for_encoder]
        return output

class ContrastiveDataset1(Dataset):
    """
    Dataset1: 从相似度高于阈值的父子评论对中构建正样本
    用于学习局部语义相似性
    """
    
    def __init__(self, post_storage: PostStorage, similarity_threshold: float = 0.5, 
                 min_subtree_size: int = 2, max_samples_per_post: Optional[int] = None):
        self.post_storage = post_storage
        self.similarity_threshold = similarity_threshold
        self.min_subtree_size = min_subtree_size
        self.positive_pairs = []
        self.comments_by_post = defaultdict(list)
        self._build_dataset(max_samples_per_post)
    
    def _ensure_str_content(self, content, default_str=""):
        if not isinstance(content, str):
            # print(f"警告: 内容不是字符串 (类型: {type(content)}), 将转换为空字符串。内容: {str(content)[:30]}...")
            return default_str
        return content

    def _build_dataset(self, max_samples_per_post: Optional[int]):
        print("🔨 构建Dataset1: 父子评论相似度对比学习数据集")
        total_pairs = 0
        for post_id, forest in tqdm(self.post_storage.forests.items(), desc="处理帖子(Dataset1)"):
            post_pairs = []
            if forest.subtrees is None: # 确保 subtrees 已初始化
                # print(f"警告: 帖子 {post_id} 的 forest.subtrees 为 None，跳过。")
                continue

            for subtree_info in forest.subtrees:
                if subtree_info['size'] >= self.min_subtree_size:
                    pairs = self._extract_high_similarity_pairs(subtree_info['root'], post_id)
                    post_pairs.extend(pairs)
            
            if max_samples_per_post and len(post_pairs) > max_samples_per_post:
                post_pairs = random.sample(post_pairs, max_samples_per_post)
            
            self.positive_pairs.extend(post_pairs)
            total_pairs += len(post_pairs)
            self._collect_comments_for_negative_sampling(forest, post_id)
        print(f"✅ Dataset1构建完成: {total_pairs} 个正样本对，覆盖 {len(self.comments_by_post)} 个帖子")

    def _extract_high_similarity_pairs(self, root, post_id) -> List[Dict]:
        pairs = []
        def traverse_node(node):
            node_content_raw = self._ensure_str_content(node.content)
            for child in node.children:
                child_content_raw = self._ensure_str_content(child.content)

                # --- 新增：预处理和过滤 ---
                parent_content_clean = preprocess_text(node_content_raw)
                child_content_clean = preprocess_text(child_content_raw)

                # 3. 当去掉这些字符后，文本长度len（）小于5的文本应该去掉
                if len(parent_content_clean) < 5 or len(child_content_clean) < 5:
                    traverse_node(child) # 即使当前对不合格，仍需继续遍历子节点
                    continue
                # --- 结束新增 ---

                try:
                    similarity = node.calculate_similarity(child) # 假设此方法能处理非字符串内容或已在内部处理
                    if similarity >= self.similarity_threshold:
                        pairs.append({
                            'parent_content': parent_content_clean, # 使用清洗后的文本
                            'child_content': child_content_clean,   # 使用清洗后的文本
                            'parent_id': node.comment_id,
                            'child_id': child.comment_id,
                            'similarity': similarity,
                            'post_id': post_id
                        })
                except (ValueError, AttributeError, TypeError) as e:
                    # print(f"警告: 计算相似度失败 (父: {node.comment_id}, 子: {child.comment_id})，错误: {e}")
                    pass
                traverse_node(child)
        if root: traverse_node(root)
        return pairs

    def _collect_comments_for_negative_sampling(self, forest, post_id):
        comments = []
        if forest.subtrees is None: return

        for subtree_info in forest.subtrees:
            def collect_from_node(node):
                content_str = self._ensure_str_content(node.content)
                comments.append({'content': content_str, 'comment_id': node.comment_id, 'post_id': post_id})
                for child in node.children:
                    collect_from_node(child)
            if subtree_info['root']: collect_from_node(subtree_info['root'])
        self.comments_by_post[post_id] = comments
    
    def __len__(self):
        return len(self.positive_pairs)
    
    def __getitem__(self, idx):
        pair = self.positive_pairs[idx]
        return {
            'anchor_content': self._ensure_str_content(pair['parent_content']),
            'positive_content': self._ensure_str_content(pair['child_content']),
            'post_id': pair['post_id'],
            'similarity_score': pair['similarity'],
            'pair_type': 'parent_child'
        }
    
    def get_negative_samples(self, post_id: str, num_negatives: int = 1) -> List[str]:
        other_posts = [pid for pid in self.comments_by_post.keys() if pid != post_id]
        if not other_posts: return []
        negative_contents = []
        attempts = 0
        max_attempts = num_negatives * 5 # 避免无限循环
        while len(negative_contents) < num_negatives and attempts < max_attempts:
            neg_post_id = random.choice(other_posts)
            neg_comments = self.comments_by_post[neg_post_id]
            if neg_comments:
                neg_comment = random.choice(neg_comments)
                content_str = self._ensure_str_content(neg_comment['content'])
                if content_str: # 确保负样本内容不为空
                    negative_contents.append(content_str)
            attempts +=1
        # 如果仍然不足，用占位符或重复
        while len(negative_contents) < num_negatives:
            negative_contents.append("<unk>") # 或者其他占位符
        return negative_contents

class ContrastiveDataset2(Dataset):
    """
    Dataset2: 从子树节点与子树平均内容构建正样本
    用于学习全局聚类相似性
    """
    def __init__(self, post_storage: PostStorage, min_subtree_size: int = 3,
                 max_samples_per_subtree: Optional[int] = None):
        self.post_storage = post_storage
        self.min_subtree_size = max(min_subtree_size, 3)
        self.positive_pairs = []
        self.comments_by_post = defaultdict(list)
        self._build_dataset(max_samples_per_subtree)

    def _ensure_str_content(self, content, default_str=""):
        if not isinstance(content, str):
            # print(f"警告: 内容不是字符串 (类型: {type(content)}), 将转换为空字符串。内容: {str(content)[:30]}...")
            return default_str
        return content

    def _build_dataset(self, max_samples_per_subtree: Optional[int]):
        print("🔨 构建Dataset2: 节点-子树中心对比学习数据集")
        total_pairs = 0
        for post_id, forest in tqdm(self.post_storage.forests.items(), desc="处理帖子(Dataset2)"):
            post_pairs = []
            if forest.subtrees is None: continue

            for subtree_info in forest.subtrees:
                if subtree_info['size'] >= self.min_subtree_size and subtree_info['root']:
                    subtree_node_contents = self._collect_subtree_node_contents(subtree_info['root'])
                    if subtree_node_contents: # 确保列表不为空
                        pairs = self._extract_node_center_pairs(
                            subtree_info['root'], subtree_node_contents, post_id,
                            max_samples_per_subtree
                        )
                        post_pairs.extend(pairs)
            
            self.positive_pairs.extend(post_pairs)
            total_pairs += len(post_pairs)
            self._collect_comments_for_negative_sampling(forest, post_id)
        print(f"✅ Dataset2构建完成: {total_pairs} 个正样本对，覆盖 {len(self.comments_by_post)} 个帖子")

    def _collect_subtree_node_contents(self, root) -> List[str]:
        contents = []
        def collect_contents(node):
            content_raw = self._ensure_str_content(node.content)
            # --- 新增：预处理和过滤 ---
            content_clean = preprocess_text(content_raw)
            if len(content_clean) >= 5: # 只添加长度合格的干净文本
                contents.append(content_clean)
            # --- 结束新增 ---
            for child in node.children:
                collect_contents(child)
        if root: collect_contents(root)
        return contents

    def _extract_node_center_pairs(self, root, subtree_node_contents: List[str], post_id,
                                 max_samples: Optional[int]) -> List[Dict]:
        pairs = []
        # subtree_node_contents 列表此时已是清洗和过滤后的，但我们仍需检查它是否为空
        if not subtree_node_contents: return []

        # 收集所有节点，即使其内容为空，因为它们仍然是子树的一部分
        # 但用于配对的 anchor_content 必须是有效字符串
        all_nodes_in_subtree = []
        def _collect_all_nodes(node):
            all_nodes_in_subtree.append(node)
            for child in node.children:
                _collect_all_nodes(child)
        if root: _collect_all_nodes(root)
        
        for node in all_nodes_in_subtree:
            node_content_raw = self._ensure_str_content(node.content)
            # --- 新增：对锚点文本进行预处理和过滤 ---
            node_content_clean = preprocess_text(node_content_raw)
            if len(node_content_clean) < 5:
                continue # 如果锚点文本不合格，则跳过
            # --- 结束新增 ---

            # 锚点内容必须有效 (此检查现在是多余的，但保留无害)
            if node_content_clean: 
                pairs.append({
                    'node_content': node_content_clean, # 使用清洗后的文本
                    'center_node_contents': subtree_node_contents, # 此列表已在收集中被清洗
                    'node_id': node.comment_id,
                    'post_id': post_id,
                    'subtree_size': len(all_nodes_in_subtree) # 使用实际收集到的节点数
                })
        
        if max_samples and len(pairs) > max_samples:
            pairs = random.sample(pairs, max_samples)
        return pairs
    
    def _collect_comments_for_negative_sampling(self, forest, post_id):
        # 与Dataset1中的方法相同
        comments = []
        if forest.subtrees is None: return
        for subtree_info in forest.subtrees:
            def collect_from_node(node):
                content_str = self._ensure_str_content(node.content)
                comments.append({'content': content_str, 'comment_id': node.comment_id, 'post_id': post_id})
                for child in node.children:
                    collect_from_node(child)
            if subtree_info['root']: collect_from_node(subtree_info['root'])
        self.comments_by_post[post_id] = comments
            
    def __len__(self):
        return len(self.positive_pairs)
    
    def __getitem__(self, idx):
        pair = self.positive_pairs[idx]
        return {
            'anchor_content': self._ensure_str_content(pair['node_content']),
            'positive_content_list': [self._ensure_str_content(c) for c in pair['center_node_contents']],
            'post_id': pair['post_id'],
            'subtree_size': pair['subtree_size'],
            'pair_type': 'node_center',
            'is_center_embedding': False
        }

    def get_negative_samples(self, post_id: str, num_negatives: int = 1) -> List[str]:
        # 与Dataset1中的方法相同
        other_posts = [pid for pid in self.comments_by_post.keys() if pid != post_id]
        if not other_posts: return []
        negative_contents = []
        attempts = 0
        max_attempts = num_negatives * 5 
        while len(negative_contents) < num_negatives and attempts < max_attempts:
            neg_post_id = random.choice(other_posts)
            neg_comments = self.comments_by_post[neg_post_id]
            if neg_comments:
                neg_comment = random.choice(neg_comments)
                content_str = self._ensure_str_content(neg_comment['content'])
                if content_str:
                    negative_contents.append(content_str)
            attempts +=1
        while len(negative_contents) < num_negatives:
            negative_contents.append("<unk>")
        return negative_contents

class ContrastiveEncoder(torch.nn.Module):
    """
    🔧 修改后的对比编码器：支持ModelScope AutoModel和自定义TextCNN。
    """
    def __init__(self, model_type: str,
                 model_name_or_path: Optional[str] = None, # ModelScope模型路径或TextCNN的名称
                 vocab: Optional[Dict[str, int]] = None, # 仅TextCNN需要
                 textcnn_config: Optional[Dict] = None, # 仅TextCNN需要
                 projection_hidden_dim: int = 512, 
                 projection_output_dim: int = 256, 
                 projection_dropout_rate: float = 0.1):
        super().__init__()
        self.model_type = model_type.lower()
        self.tokenizer = None # 初始化
        self.base_model = None # 初始化
        self.base_dim = 0 # 初始化

        if self.model_type == 'modelscope':
            if model_name_or_path is None:
                raise ValueError("对于 'modelscope' 模型类型，model_name_or_path 是必需的")
            
            # 使用ModelScope加载模型和分词器
            self.tokenizer = load_tokenizer_from_modelscope(model_name_or_path)
            self.base_model = load_model_from_modelscope(
                model_name_or_path,
                torch_dtype=torch.float32  # 强制float32
            )
            
            if hasattr(self.base_model.config, 'hidden_size'):
                 self.base_dim = self.base_model.config.hidden_size
            elif hasattr(self.base_model.config, 'd_model'): # 例如T5
                 self.base_dim = self.base_model.config.d_model
            else:
                # 尝试从模型输出获取维度（如果模型已加载参数）
                try:
                    dummy_input = self.tokenizer("test", return_tensors="pt")
                    # 移除 token_type_ids 如果模型不支持 (例如 distilbert)
                    if 'token_type_ids' in dummy_input and not any(p.name == 'token_type_ids' for p in self.base_model.forward.__code__.co_varnames):
                        del dummy_input['token_type_ids']
                    dummy_output = self.base_model(**dummy_input)
                    if hasattr(dummy_output, 'last_hidden_state'):
                        self.base_dim = dummy_output.last_hidden_state.shape[-1]
                    elif isinstance(dummy_output, torch.Tensor):
                         self.base_dim = dummy_output.shape[-1]
                    else:
                        raise ValueError(f"无法自动确定ModelScope模型 {model_name_or_path} 的基础维度。请检查模型配置。")
                except Exception as e:
                     raise ValueError(f"尝试确定ModelScope模型 {model_name_or_path} 基础维度时出错: {e}")

            print(f"🏗️ ModelScope ContrastiveEncoder 初始化完成:")
            print(f"   基础模型: {model_name_or_path}")

        elif self.model_type == 'textcnn':
            if vocab is None or textcnn_config is None:
                raise ValueError("对于 'textcnn' 模型类型，vocab 和 textcnn_config 是必需的")
            
            _max_len = textcnn_config.get('max_seq_length', 128)
            self.tokenizer = TextCNNTokenizer(vocab, max_length=_max_len)
            self.base_model = TextCNNModel(
                vocab_size=len(vocab),
                embedding_dim=textcnn_config['embedding_dim'],
                num_filters=textcnn_config['num_filters'],
                filter_sizes=textcnn_config['filter_sizes'],
                output_dim_for_encoder=textcnn_config['textcnn_output_dim'],
                dropout_rate=textcnn_config.get('model_dropout_rate', 0.1)
            )
            self.base_dim = textcnn_config['textcnn_output_dim']
            print(f"🏗️ TextCNN ContrastiveEncoder 初始化完成:")
            print(f"   TextCNN 词汇表大小: {len(vocab)}")
            print(f"   TextCNN 配置: {textcnn_config}")
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}。请选择 'modelscope' 或 'textcnn'。")

        # 投影头 (两种模型类型通用)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(self.base_dim, projection_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(projection_dropout_rate),
            torch.nn.Linear(projection_hidden_dim, projection_output_dim),
            torch.nn.LayerNorm(projection_output_dim)
        ).float() # 确保投影头是 float32
        
        print(f"   基础模型输出维度 (到投影头): {self.base_dim}")
        print(f"   投影头输入维度: {self.base_dim}")
        print(f"   投影头隐藏层维度: {projection_hidden_dim}")
        print(f"   投影头输出维度 (最终嵌入): {projection_output_dim}")
        if self.base_model:
             print(f"   基础模型数据类型: {next(self.base_model.parameters()).dtype}")
        print(f"   投影头数据类型: {next(self.projection_head.parameters()).dtype}")

    def _ensure_float32(self, tensor):
        if tensor.dtype in [torch.bfloat16, torch.float16]:
            return tensor.float()
        return tensor

    def _ensure_list_of_strings(self, texts: Union[str, List[any]]) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = []
        for text_input in texts:
            if not isinstance(text_input, str):
                # 处理 None 或其他类型，将其转换为空字符串
                processed_texts.append(str(text_input) if text_input is not None else "")
            else:
                processed_texts.append(text_input if text_input else "") # 确保空输入也是空字符串
        return processed_texts

    def get_base_embeddings(self, texts: Union[str, List[any]]):
        """
        获取基础模型的嵌入（在投影头之前）。
        """
        processed_texts = self._ensure_list_of_strings(texts)
        if not processed_texts:
            return torch.empty(0, self.base_dim, device=next(self.parameters()).device)

        device = next(self.parameters()).device
        
        max_len = 512
        if self.model_type == 'textcnn':
            max_len = self.tokenizer.max_length
        elif hasattr(self.tokenizer, 'model_max_length'):
            max_len = self.tokenizer.model_max_length

        inputs = self.tokenizer(
            processed_texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=max_len
        )
        # 修复: 将字典中的每个张量移动到设备上
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = self.base_model(**inputs)

        if self.model_type == 'modelscope':
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden = self._ensure_float32(outputs.last_hidden_state)
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                base_embeddings = sum_embeddings / sum_mask
            elif hasattr(outputs, 'pooler_output'):
                base_embeddings = self._ensure_float32(outputs.pooler_output)
            elif isinstance(outputs, torch.Tensor):
                base_embeddings = self._ensure_float32(outputs)
            else:
                raise ValueError("无法从ModelScope模型输出确定基础嵌入。")
        elif self.model_type == 'textcnn':
            base_embeddings = self._ensure_float32(outputs)
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")

        return base_embeddings

    def forward(self, texts: Union[str, List[any]]):
        """
        定义模型的完整前向传播：基础模型 -> 投影头。
        """
        base_embeddings = self.get_base_embeddings(texts)
        
        if base_embeddings.nelement() == 0:
            # 从投影头的最后一层获取输出维度
            proj_output_dim = self.projection_head[-2].out_features
            return torch.empty(0, proj_output_dim, device=base_embeddings.device, dtype=torch.float)
            
        projected_embeddings = self.projection_head(base_embeddings)
        return projected_embeddings

    def save_base_model(self, path: str):
        """保存基础模型 (ModelScope 或 TextCNN) 及其分词器/词汇表。"""
        os.makedirs(path, exist_ok=True)
        if self.model_type == 'modelscope':
            # 对于ModelScope模型，保存方式与HuggingFace类似
            self.base_model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"💾 ModelScope 基础模型和分词器已保存到: {path}")
        elif self.model_type == 'textcnn':
            model_save_path = os.path.join(path, "textcnn_model.pth")
            vocab_save_path = os.path.join(path, "textcnn_vocab.json")
            tokenizer_config_path = os.path.join(path, "textcnn_tokenizer_config.json")

            torch.save(self.base_model.state_dict(), model_save_path)
            with open(vocab_save_path, 'w', encoding='utf-8') as f:
                json.dump(self.tokenizer.word_to_idx, f, ensure_ascii=False, indent=4)
            
            tokenizer_config = {'max_length': self.tokenizer.max_length}
            with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_config, f, ensure_ascii=False, indent=4)

            print(f"💾 TextCNN 基础模型 state_dict 已保存到: {model_save_path}")
            print(f"💾 TextCNN 词汇表已保存到: {vocab_save_path}")
            print(f"💾 TextCNN 分词器配置已保存到: {tokenizer_config_path}")
        else:
            print(f"⚠️ 未知模型类型 '{self.model_type}'，无法保存基础模型。")

class ContrastiveDataCollator:
    """
    自定义的DataCollator，用于批量处理并动态添加负样本
    """
    def __init__(self, dataset: Union[ContrastiveDataset1, ContrastiveDataset2], num_negatives: int = 2):
        self.dataset = dataset
        self.num_negatives = num_negatives
    
    def _ensure_str_content(self, content, default_str="<unk>"): # Collator中的默认值
        if not isinstance(content, str):
            # print(f"警告 (Collator): 内容不是字符串 (类型: {type(content)}), 将转换为 '{default_str}'。内容: {str(content)[:30]}...")
            return default_str
        return content if content else default_str # 空字符串也转为默认值

    def __call__(self, batch: List[Dict]) -> Dict[str, any]:
        anchor_texts = [self._ensure_str_content(item['anchor_content']) for item in batch]
        
        positive_texts_ds1 = [] 
        positive_content_lists_ds2 = []

        for item in batch:
            pair_type = item.get('pair_type', 'unknown')
            if pair_type == 'parent_child':
                positive_texts_ds1.append(self._ensure_str_content(item['positive_content']))
                positive_content_lists_ds2.append(None)
            elif pair_type == 'node_center':
                positive_texts_ds1.append(None)
                # 确保列表中的每个内容都是字符串
                content_list = item.get('positive_content_list', [])
                if isinstance(content_list, list):
                    processed_list = [self._ensure_str_content(c) for c in content_list]
                    positive_content_lists_ds2.append(processed_list if processed_list else [self._ensure_str_content("")]) # 保证不为空列表
                else: # 如果不是列表，则创建一个包含单个元素的列表
                    positive_content_lists_ds2.append([self._ensure_str_content(content_list)])

            else: # 默认或未知类型
                positive_texts_ds1.append(self._ensure_str_content(item.get('positive_content', item['anchor_content'])))
                positive_content_lists_ds2.append(None)
        
        negative_texts = []
        if self.num_negatives > 0:
            for item in batch:
                post_id = item['post_id']
                neg_contents = []
                if hasattr(self.dataset, 'get_negative_samples'):
                    neg_contents = self.dataset.get_negative_samples(post_id, self.num_negatives)
                
                # 确保负样本是字符串且数量正确
                processed_neg_contents = []
                for neg_c in neg_contents:
                    processed_neg_contents.append(self._ensure_str_content(neg_c))
                
                while len(processed_neg_contents) < self.num_negatives:
                    # 尝试从批内其他帖子获取
                    other_items_texts = [self._ensure_str_content(b['anchor_content']) for b in batch if b['post_id'] != post_id and self._ensure_str_content(b['anchor_content'])]
                    if other_items_texts:
                        processed_neg_contents.append(random.choice(other_items_texts))
                    else: # 万不得已用占位符
                        processed_neg_contents.append(self._ensure_str_content("")) 
                
                negative_texts.extend(processed_neg_contents[:self.num_negatives])
        
        return {
            'anchor_texts': anchor_texts,
            'positive_texts_ds1': positive_texts_ds1, 
            'positive_content_lists_ds2': positive_content_lists_ds2, 
            'negative_texts': negative_texts, # 扁平列表
            'post_ids': [item['post_id'] for item in batch],
            'pair_types': [item.get('pair_type', 'unknown') for item in batch],
            'num_negatives': self.num_negatives
        }

class ContrastiveLoss(torch.nn.Module):
    """
    对比损失函数，支持三种InfoNCE变体
    """
    def __init__(self, temperature: float = 0.07, loss_type: str = 'infonce', 
                 infonce_mode: str = 'unidirectional'):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.infonce_mode = infonce_mode
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        print(f"🎯 ContrastiveLoss配置: 类型={loss_type}, InfoNCE模式={infonce_mode}, 温度={temperature}")
        
    def forward(self, anchor, positive, negatives=None):
        if anchor.nelement() == 0 or positive.nelement() == 0: # 处理空输入
            return torch.tensor(0.0, device=anchor.device if anchor.nelement() > 0 else positive.device, requires_grad=True)

        if self.loss_type == 'infonce':
            if self.infonce_mode == 'unidirectional':
                return self._infonce_loss_unidirectional(anchor, positive, negatives)
            elif self.infonce_mode == 'bidirectional':
                return self._infonce_loss_bidirectional(anchor, positive, negatives)
            elif self.infonce_mode == 'in_batch':
                return self._infonce_loss_in_batch(anchor, positive)
            else:
                raise ValueError(f"不支持的 InfoNCE 模式: {self.infonce_mode}")
        # ... (triplet loss can be added here if needed)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
    
    def _infonce_loss_unidirectional(self, anchor, positive, negatives):
        if negatives is None or negatives.nelement() == 0: # 检查 negatives 是否有效
            # 如果没有负样本，可以考虑退化为简单的正样本匹配或返回0损失
            # print("警告: 单向InfoNCE需要有效的负样本，但未提供或为空。返回0损失。")
            # 计算正样本相似度，但不进行对比损失
            # pos_sim = self.cosine_sim(anchor, positive) / self.temperature
            # return -pos_sim.mean() # 尝试最大化正样本相似度，但这已不是InfoNCE
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)

        batch_size, num_negatives, _ = negatives.shape
        
        anchor_norm = F.normalize(anchor, dim=-1)
        positive_norm = F.normalize(positive, dim=-1)
        negatives_norm = F.normalize(negatives, dim=-1)
        
        pos_sim = self.cosine_sim(anchor_norm, positive_norm) / self.temperature
        
        anchor_expanded = anchor_norm.unsqueeze(1).expand(-1, num_negatives, -1)
        neg_sim = self.cosine_sim(anchor_expanded, negatives_norm) / self.temperature
        
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(all_sim, labels)
        return loss
    
    def _infonce_loss_bidirectional(self, anchor, positive, negatives):
        if negatives is None or negatives.nelement() == 0:
            # print("警告: 双向InfoNCE需要有效的负样本，但未提供或为空。返回0损失。")
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)
            
        loss1 = self._compute_single_direction_loss(anchor, positive, negatives)
        loss2 = self._compute_single_direction_loss(positive, anchor, negatives)
        return (loss1 + loss2) / 2.0
    
    def _infonce_loss_in_batch(self, anchor, positive):
        batch_size = anchor.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)
        
        anchor_norm = F.normalize(anchor, dim=-1)
        positive_norm = F.normalize(positive, dim=-1)
        
        # 正样本对的相似度 (anchor_i 与 positive_i)
        # 对于in-batch，我们通常假设 anchor 和 positive 来自同一组数据，只是增强不同
        # 或者 anchor[i] 的正样本是 positive[i]
        # logits 分子: sim(anchor_i, positive_i)
        # logits 分母: sim(anchor_i, positive_i) + sum_{j!=i} sim(anchor_i, positive_j)
        
        # 方法1: anchor[i] vs positive[j] for all j. Positive is positive[i]
        similarity_matrix_ap = torch.matmul(anchor_norm, positive_norm.t()) / self.temperature # [B, B]
        labels_ap = torch.arange(batch_size, device=anchor.device)
        loss_ap = F.cross_entropy(similarity_matrix_ap, labels_ap)
        
        # 方法2: positive[i] vs anchor[j] for all j. Positive is anchor[i]
        # 这等价于上面的转置，但为了概念清晰
        similarity_matrix_pa = torch.matmul(positive_norm, anchor_norm.t()) / self.temperature # [B, B]
        labels_pa = torch.arange(batch_size, device=anchor.device)
        loss_pa = F.cross_entropy(similarity_matrix_pa, labels_pa)

        return (loss_ap + loss_pa) / 2.0

    def _compute_single_direction_loss(self, query, positive_key, negative_keys):
        # query: [B, D], positive_key: [B, D], negative_keys: [B, N, D]
        batch_size, num_negatives, _ = negative_keys.shape

        query_norm = F.normalize(query, dim=-1)
        positive_key_norm = F.normalize(positive_key, dim=-1)
        negative_keys_norm = F.normalize(negative_keys, dim=-1)

        pos_sim = self.cosine_sim(query_norm, positive_key_norm) / self.temperature # [B]
        
        query_expanded = query_norm.unsqueeze(1).expand(-1, num_negatives, -1) # [B, N, D]
        neg_sim = self.cosine_sim(query_expanded, negative_keys_norm) / self.temperature # [B, N]
        
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) # [B, 1+N]
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        loss = F.cross_entropy(all_sim, labels)
        return loss

class DynamicContrastiveTrainer:
    def __init__(self, post_storage: PostStorage,
                 training_model_type: str = 'modelscope',
                 training_model_identifier_or_path: Optional[str] = "google-bert/bert-base-chinese",
                 textcnn_config: Optional[Dict] = None,
                 projection_head_config: Optional[Dict] = None,
                 use_peft: bool = False,  # <--- 新增参数：是否使用PEFT
                 peft_config: Optional[Dict] = None, # <--- 新增参数：PEFT配置
                 pruning_model_path: str = "google-bert/bert-base-chinese",
                 similarity_threshold: float = 0.5,
                 num_negatives: int = 2,
                 batch_size: int = 32,
                 pruning_inference_batch_size: int = 32,
                 base_lr: float = 1e-6,
                 projection_lr: float = 1e-4,
                 use_weighted_loss: bool = False,
                 loss_weights: Optional[Dict[str, float]] = None,
                 adaptive_weighting: bool = False,
                 infonce_mode: str = 'unidirectional',
                 min_subtree_size_ds1: int = 2,
                 max_samples_per_post_ds1: Optional[int] = None,
                 min_subtree_size_ds2: int = 4,
                 max_samples_per_subtree_ds2: Optional[int] = None):

        self.post_storage = post_storage
        self.training_model_type = training_model_type.lower()
        self.training_model_identifier_or_path = training_model_identifier_or_path
        self.textcnn_config = textcnn_config
        self.use_peft = use_peft
        self.peft_config = peft_config
        self.pruning_model_path = pruning_model_path
        self.similarity_threshold = similarity_threshold
        self.num_negatives = num_negatives if infonce_mode != 'in_batch' else 0
        self.batch_size = batch_size
        self.pruning_inference_batch_size = pruning_inference_batch_size
        self.infonce_mode = infonce_mode

        self.min_subtree_size_ds1 = min_subtree_size_ds1
        self.max_samples_per_post_ds1 = max_samples_per_post_ds1
        self.min_subtree_size_ds2 = min_subtree_size_ds2
        self.max_samples_per_subtree_ds2 = max_samples_per_subtree_ds2

        self.use_weighted_loss = use_weighted_loss
        if self.use_weighted_loss:
            if loss_weights is None: self.loss_weights = {'dataset1': 0.5, 'dataset2': 0.5}
            else: self.loss_weights = loss_weights.copy()
            weight_sum = sum(self.loss_weights.values())
            if weight_sum > 0:
                for k_lw in self.loss_weights: self.loss_weights[k_lw] /= weight_sum
            else:
                self.loss_weights = {'dataset1': 0.5, 'dataset2': 0.5}
            self.adaptive_weighting = adaptive_weighting
            self.initial_loss_weights = self.loss_weights.copy()
            print(f"🎯 启用损失加权: {self.loss_weights}, 自适应: {self.adaptive_weighting}")
        else:
            self.loss_weights = {'dataset1': 1.0, 'dataset2': 1.0}
            self.adaptive_weighting = False
            print("📊 使用独立损失（每个数据集的损失乘以其权重，默认为1.0）")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _default_proj_config = {'hidden_dim': 512, 'output_dim': 256, 'dropout_rate': 0.1}
        self.projection_config = {**_default_proj_config, **(projection_head_config if projection_head_config is not None else {})}


        self.vocab = None
        if self.training_model_type == 'textcnn':
            print("🛠️ 使用 TextCNN 模型进行训练。")
            if self.textcnn_config is None:
                raise ValueError("当 training_model_type 为 'textcnn' 时，textcnn_config 是必需的")
            print("🏗️ 从 PostStorage 为 TextCNN 构建词汇表...")
            self.vocab, _ = build_vocab_from_post_storage(self.post_storage, min_freq=self.textcnn_config.get('min_vocab_freq', 1))
            self.contrastive_encoder = ContrastiveEncoder(
                model_type='textcnn',
                vocab=self.vocab,
                textcnn_config=self.textcnn_config,
                projection_hidden_dim=self.projection_config['hidden_dim'],
                projection_output_dim=self.projection_config['output_dim'],
                projection_dropout_rate=self.projection_config['dropout_rate']
            ).to(self.device)
            print(f"   TextCNN 模型名称 (标识符): {self.training_model_identifier_or_path}")
        elif self.training_model_type == 'modelscope':
            print(f"🛠️ 使用 ModelScope 模型进行训练: {self.training_model_identifier_or_path}")
            if self.training_model_identifier_or_path is None:
                 raise ValueError("对于 'modelscope' 模型类型，training_model_identifier_or_path 是必需的。")
            self.contrastive_encoder = ContrastiveEncoder(
                model_type='modelscope',
                model_name_or_path=self.training_model_identifier_or_path,
                projection_hidden_dim=self.projection_config['hidden_dim'],
                projection_output_dim=self.projection_config['output_dim'],
                projection_dropout_rate=self.projection_config['dropout_rate']
            ).to(self.device)
             # --- 新增：应用PEFT/LoRA的逻辑 ---
            if self.use_peft:
                print("🚀 应用PEFT (LoRA)到基础模型...")
                
                # 定义默认LoRA配置，并与用户传入的配置合并
                default_lora_config = {
                    'r': 16,
                    'lora_alpha': 32,
                    'target_modules': ["query", "key", "value"],
                    'lora_dropout': 0.05,
                    'bias': "none",
                }
                final_lora_config_dict = {**default_lora_config, **(self.peft_config or {})}
                
                lora_config = LoraConfig(**final_lora_config_dict)
                
                # 将基础模型包装成PeftModel
                self.contrastive_encoder.base_model = get_peft_model(self.contrastive_encoder.base_model, lora_config)
                
                print("✅ LoRA应用完成。可训练参数详情:")
                self.contrastive_encoder.base_model.print_trainable_parameters()
            # --- PEFT逻辑结束 ---
        else:
            raise ValueError(f"不支持的训练模型类型: {self.training_model_type}。请选择 'modelscope' 或 'textcnn'。")

        print(f"🔍 初始化剪枝模型: {self.pruning_model_path}")
        # 使用为ModelScope定义的辅助函数加载剪枝模型
        self.pruning_tokenizer = load_tokenizer_from_modelscope(self.pruning_model_path)
        self.pruning_model = load_model_from_modelscope(
            self.pruning_model_path,
            torch_dtype=torch.float32
        )
        self.pruning_model_on_gpu = False

        self.optimizer = torch.optim.AdamW([
            {'params': self.contrastive_encoder.base_model.parameters(), 'lr': base_lr, 'weight_decay': 1e-6, 'initial_lr': base_lr},
            {'params': self.contrastive_encoder.projection_head.parameters(), 'lr': projection_lr, 'weight_decay': 1e-4, 'initial_lr': projection_lr}
        ])

        self.loss_fn1 = ContrastiveLoss(temperature=0.07, loss_type='infonce', infonce_mode=infonce_mode)
        self.loss_fn2 = ContrastiveLoss(temperature=0.05, loss_type='infonce', infonce_mode=infonce_mode)

        self.training_history = defaultdict(list)
        self.training_history['dataset_sizes'] = {'dataset1': [], 'dataset2': []}
        if self.use_weighted_loss:
            self.training_history['weight_ds1'] = []
            self.training_history['weight_ds2'] = []
        print(f"🚀 DynamicContrastiveTrainer 初始化完成。设备: {self.device}")

    def _load_pruning_model_to_gpu(self):
        if not self.pruning_model_on_gpu and self.device.type == 'cuda':
            print("🔧 将剪枝模型加载到GPU...")
            self.pruning_model.to(self.device)
            self.pruning_model_on_gpu = True

    def _unload_pruning_model_from_gpu(self):
        if self.pruning_model_on_gpu and self.device.type == 'cuda':
            print("🔧 从GPU卸载剪枝模型...")
            self.pruning_model.to('cpu')
            self.pruning_model_on_gpu = False
            torch.cuda.empty_cache()

    def _get_pruning_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        # self._load_pruning_model_to_gpu() # 由调用者管理GPU加载/卸载
        texts = [str(t) if not isinstance(t, str) else t for t in texts]
        texts = [t if t else "<empty_text_placeholder>" for t in texts] # 替换空字符串

        all_embeddings_list = []
        self.pruning_model.eval()
        with torch.no_grad():
            # 使用tqdm包裹批处理循环
            for i in tqdm(range(0, len(texts), self.pruning_inference_batch_size), desc="计算剪枝嵌入", unit="batch"):
                batch_texts = texts[i:i + self.pruning_inference_batch_size]
                if not batch_texts: continue
                try:
                    inputs = self.pruning_tokenizer(
                        batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512
                    ).to(self.pruning_model.device)
                    outputs = self.pruning_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    if batch_embeddings.dtype in [torch.bfloat16, torch.float16]:
                        batch_embeddings = batch_embeddings.float()
                    all_embeddings_list.append(batch_embeddings.cpu())
                except Exception as e:
                    print(f"获取剪枝嵌入时批处理错误: {e}. 跳过此批次。批次文本示例: {batch_texts[0][:50] if batch_texts else 'N/A'}")
                    # 根据需要，可以为失败的批次添加占位符嵌入，或者简单地跳过
                    # 例如，添加零嵌入：
                    # output_dim = self.pruning_model.config.hidden_size # 假设HF模型
                    # all_embeddings_list.append(torch.zeros((len(batch_texts), output_dim), dtype=torch.float32).cpu())


        if not all_embeddings_list: return np.array([])
        try:
            all_embeddings_np = torch.cat(all_embeddings_list, dim=0).numpy()
        except RuntimeError as e:
            print(f"连接剪枝嵌入时出错: {e}")
            # 尝试找出哪个嵌入导致问题
            # for i, emb_tensor in enumerate(all_embeddings_list):
            # print(f"Tensor {i} shape: {emb_tensor.shape}, dtype: {emb_tensor.dtype}")
            return np.array([]) # 返回空数组以避免进一步错误
        return all_embeddings_np

    def _build_and_log_datasets(self):
        print("🛠️ 构建/重建数据集...")
        print("   收集所有评论用于剪枝模型嵌入...")
        all_comment_texts = []
        comment_nodes_references = []

        for post_id, post_container in self.post_storage.posts.items():
            # Path A: 尝试从 'comments' 字典获取 (如果存在且被填充)
            if hasattr(post_container, 'comments') and isinstance(post_container.comments, dict) and post_container.comments:
                # print(f"DEBUG: Post {post_id} - 使用 'comments' 字典路径找到 {len(post_container.comments)} 条评论。")
                for comment_id, comment_node in post_container.comments.items():
                    content_str = str(comment_node.content) if comment_node.content is not None else ""
                    all_comment_texts.append(content_str)
                    comment_nodes_references.append(comment_node)
            # Path B: 尝试从 'root' 属性遍历 (与 build_vocab_from_post_storage 一致)
            elif hasattr(post_container, 'root') and post_container.root:
                # print(f"DEBUG: Post {post_id} - 使用 'root' 属性路径。")
                queue = [post_container.root] # 使用 root
                visited_ids_in_post = set()
                while queue:
                    comment_node = queue.pop(0)
                    if comment_node.comment_id in visited_ids_in_post:
                        continue
                    visited_ids_in_post.add(comment_node.comment_id)
                    
                    content_str = str(comment_node.content) if comment_node.content is not None else ""
                    all_comment_texts.append(content_str)
                    comment_nodes_references.append(comment_node)
                    
                    if hasattr(comment_node, 'children') and comment_node.children: # 使用 children
                        for child_node in comment_node.children: # 使用 children
                            if child_node:
                                queue.append(child_node)
            # else:
                # print(f"DEBUG: Post {post_id} - 未找到 'comments' 字典或 'root' 属性。")

        if all_comment_texts:
            # --- 新增：在计算剪枝嵌入前，对文本进行预处理 ---
            print(f"   对 {len(all_comment_texts)} 条评论进行预处理以用于剪枝...")
            preprocessed_texts_for_pruning = [preprocess_text(text) for text in all_comment_texts]

            print(f"   计算 {len(preprocessed_texts_for_pruning)} 条预处理后评论的剪枝嵌入...")
            self._load_pruning_model_to_gpu()
            pruning_embeddings_np = self._get_pruning_embeddings(preprocessed_texts_for_pruning)
            self._unload_pruning_model_from_gpu()

            if pruning_embeddings_np.ndim == 2 and pruning_embeddings_np.shape[0] == len(comment_nodes_references):
                for i, comment_node in enumerate(comment_nodes_references):
                    if hasattr(comment_node, 'set_embedding'):
                        comment_node.set_embedding(pruning_embeddings_np[i])
                    else: # 后备方案
                        comment_node.embedding = pruning_embeddings_np[i]
                print("   剪枝嵌入已为PostStorage中的所有评论设置。")
            else:
                print(f"   警告: _get_pruning_embeddings 的形状不匹配或结果为空。预期 ({len(comment_nodes_references)}, dim)，得到 {pruning_embeddings_np.shape if isinstance(pruning_embeddings_np, np.ndarray) else 'Not an ndarray'}。跳过剪枝嵌入更新。")
        else:
            print("   在PostStorage中未找到评论进行剪枝嵌入。")

        build_pruned_forest(self.post_storage, self.similarity_threshold)

        print("   创建 Dataset1...")
        self.dataset1 = ContrastiveDataset1(
            post_storage=self.post_storage,
            similarity_threshold=self.similarity_threshold,
            min_subtree_size=self.min_subtree_size_ds1,
            max_samples_per_post=self.max_samples_per_post_ds1
        )
        print("   创建 Dataset2...")
        self.dataset2 = ContrastiveDataset2(
            post_storage=self.post_storage,
            min_subtree_size=self.min_subtree_size_ds2,
            max_samples_per_subtree=self.max_samples_per_subtree_ds2
        )

        ds1_size = len(self.dataset1)
        ds2_size = len(self.dataset2)
        self.training_history['dataset_sizes']['dataset1'].append(ds1_size)
        self.training_history['dataset_sizes']['dataset2'].append(ds2_size)
        print(f"   Dataset1 大小: {ds1_size}, Dataset2 大小: {ds2_size}")

        if ds1_size == 0 and ds2_size == 0:
            print("⚠️ 警告: 两个数据集都为空。训练可能无法有效进行。")

        collator_num_neg = self.num_negatives
        self.collator1 = ContrastiveDataCollator(self.dataset1, num_negatives=collator_num_neg)
        self.collator2 = ContrastiveDataCollator(self.dataset2, num_negatives=collator_num_neg)

        self.train_loader1 = DataLoader(self.dataset1, batch_size=self.batch_size, shuffle=True, collate_fn=self.collator1, num_workers=0, pin_memory=True if self.device.type == 'cuda' else False) if ds1_size > 0 else None
        self.train_loader2 = DataLoader(self.dataset2, batch_size=self.batch_size, shuffle=True, collate_fn=self.collator2, num_workers=0, pin_memory=True if self.device.type == 'cuda' else False) if ds2_size > 0 else None
        print("✅ 数据集和 DataLoader 已准备就绪。")

        # --- 新增代码：保存构建好的数据集 ---
        print("💾 正在保存构建好的数据集...")
        save_dir = "cl_dataset"
        os.makedirs(save_dir, exist_ok=True)

        # 创建动态文件名
        sanitized_model_name = self.training_model_identifier_or_path.replace('/', '_')
        similarity_str = str(self.similarity_threshold)
        base_filename = f"{sanitized_model_name}_sim_{similarity_str}"

        # 保存 Dataset1
        if ds1_size > 0:
            ds1_filepath = os.path.join(save_dir, f"{base_filename}_dataset1.pkl")
            try:
                with open(ds1_filepath, 'wb') as f:
                    pickle.dump(self.dataset1, f)
                print(f"   -> Dataset1 已保存至: {ds1_filepath}")
            except Exception as e:
                print(f"   -> ❌ 保存 Dataset1 失败: {e}")
        else:
            print("   -> Dataset1 为空，不进行保存。")

        # 保存 Dataset2
        if ds2_size > 0:
            ds2_filepath = os.path.join(save_dir, f"{base_filename}_dataset2.pkl")
            try:
                with open(ds2_filepath, 'wb') as f:
                    pickle.dump(self.dataset2, f)
                print(f"   -> Dataset2 已保存至: {ds2_filepath}")
            except Exception as e:
                print(f"   -> ❌ 保存 Dataset2 失败: {e}")
        else:
            print("   -> Dataset2 为空，不进行保存。")

    def _process_batch(self, batch: Dict, loss_fn: ContrastiveLoss, dataset_name: str) -> Optional[torch.Tensor]:
        self.optimizer.zero_grad()

        anchor_texts = batch['anchor_texts']
        positive_texts_ds1 = batch.get('positive_texts_ds1', [])
        positive_content_lists_ds2 = batch.get('positive_content_lists_ds2', [])
        negative_texts_flat = batch.get('negative_texts', []) # 扁平列表
        num_negatives_per_anchor = batch.get('num_negatives', 0)

        if not anchor_texts: return None
        loss = None
        processed_anchors_count = 0 # 当前批次中实际处理的锚点数量

        if dataset_name == 'dataset1':
            valid_indices = [i for i, txt in enumerate(positive_texts_ds1) if txt is not None]
            if not valid_indices: return None

            current_anchor_texts = [anchor_texts[i] for i in valid_indices]
            current_positive_texts = [positive_texts_ds1[i] for i in valid_indices]
            processed_anchors_count = len(current_anchor_texts)
            if processed_anchors_count == 0: return None


            anchor_emb = self.contrastive_encoder(current_anchor_texts)
            positive_emb = self.contrastive_encoder(current_positive_texts)

            neg_emb_reshaped = None
            if self.infonce_mode != 'in_batch' and num_negatives_per_anchor > 0 and negative_texts_flat:
                # 从扁平列表中提取与 valid_indices 对应的负样本
                current_negative_texts_for_batch = []
                for original_batch_idx in valid_indices:
                    start_idx_flat = original_batch_idx * num_negatives_per_anchor
                    end_idx_flat = start_idx_flat + num_negatives_per_anchor
                    current_negative_texts_for_batch.extend(negative_texts_flat[start_idx_flat:end_idx_flat])

                if current_negative_texts_for_batch:
                    neg_emb_flat = self.contrastive_encoder(current_negative_texts_for_batch)
                    if neg_emb_flat.nelement() > 0:
                        try:
                            neg_emb_reshaped = neg_emb_flat.view(processed_anchors_count, num_negatives_per_anchor, -1)
                        except RuntimeError as e:
                            print(f"DS1 Reshape error: {e}. Anchors: {processed_anchors_count}, Neg per anchor: {num_negatives_per_anchor}, Flat neg shape: {neg_emb_flat.shape}")
                            return None # or handle differently

            if anchor_emb.nelement() > 0 and positive_emb.nelement() > 0:
                loss = loss_fn(anchor_emb, positive_emb, neg_emb_reshaped if neg_emb_reshaped is not None and neg_emb_reshaped.nelement() > 0 else None)

        elif dataset_name == 'dataset2':
            valid_indices = [i for i, lst in enumerate(positive_content_lists_ds2) if lst is not None and any(s and s.strip() for s in lst)]
            if not valid_indices: return None

            current_anchor_texts = [anchor_texts[i] for i in valid_indices]
            current_positive_lists = [positive_content_lists_ds2[i] for i in valid_indices]
            processed_anchors_count = len(current_anchor_texts)
            if processed_anchors_count == 0: return None

            anchor_emb = self.contrastive_encoder(current_anchor_texts)
            positive_emb_list_for_ds2 = []
            for text_list_for_one_anchor in current_positive_lists:
                valid_texts_in_list = [s for s in text_list_for_one_anchor if s and s.strip()]
                if valid_texts_in_list:
                    # 获取基础嵌入，然后平均，然后投影
                    node_base_embeddings = self.contrastive_encoder.get_base_embeddings(valid_texts_in_list)
                    if node_base_embeddings.nelement() > 0:
                        avg_node_base_embedding = node_base_embeddings.mean(dim=0, keepdim=True)
                        projected_avg_emb = self.contrastive_encoder.projection_head(avg_node_base_embedding)
                        positive_emb_list_for_ds2.append(projected_avg_emb)
                    else: # 如果列表中的所有文本都无效/空，则添加零嵌入
                        proj_output_dim = self.contrastive_encoder.projection_head[-2].out_features
                        positive_emb_list_for_ds2.append(torch.zeros((1, proj_output_dim), device=self.device, dtype=anchor_emb.dtype))
                else: # 如果整个列表为空或仅包含无效字符串
                    proj_output_dim = self.contrastive_encoder.projection_head[-2].out_features
                    positive_emb_list_for_ds2.append(torch.zeros((1, proj_output_dim), device=self.device, dtype=anchor_emb.dtype))

            if not positive_emb_list_for_ds2 or len(positive_emb_list_for_ds2) != processed_anchors_count:
                 print(f"DS2: positive_emb_list 长度 ({len(positive_emb_list_for_ds2)}) 与锚点数量 ({processed_anchors_count}) 不匹配。")
                 return None
            positive_emb = torch.cat(positive_emb_list_for_ds2, dim=0)

            neg_emb_reshaped = None
            if self.infonce_mode != 'in_batch' and num_negatives_per_anchor > 0 and negative_texts_flat:
                current_negative_texts_for_batch = []
                for original_batch_idx in valid_indices:
                    start_idx_flat = original_batch_idx * num_negatives_per_anchor
                    end_idx_flat = start_idx_flat + num_negatives_per_anchor
                    current_negative_texts_for_batch.extend(negative_texts_flat[start_idx_flat:end_idx_flat])

                if current_negative_texts_for_batch:
                    neg_emb_flat = self.contrastive_encoder(current_negative_texts_for_batch)
                    if neg_emb_flat.nelement() > 0:
                        try:
                            neg_emb_reshaped = neg_emb_flat.view(processed_anchors_count, num_negatives_per_anchor, -1)
                        except RuntimeError as e:
                            print(f"DS2 Reshape error: {e}. Anchors: {processed_anchors_count}, Neg per anchor: {num_negatives_per_anchor}, Flat neg shape: {neg_emb_flat.shape}")
                            return None


            if anchor_emb.nelement() > 0 and positive_emb.nelement() > 0:
                loss = loss_fn(anchor_emb, positive_emb, neg_emb_reshaped if neg_emb_reshaped is not None and neg_emb_reshaped.nelement() > 0 else None)

        if loss is not None and loss.requires_grad:
            effective_weight = self.loss_weights.get(dataset_name, 1.0) if self.use_weighted_loss else 1.0
            (loss * effective_weight).backward()
            torch.nn.utils.clip_grad_norm_(self.contrastive_encoder.parameters(), max_norm=1.0)
            self.optimizer.step()
            return loss # 返回未缩放的损失用于记录
        return None

    def train(self, num_epochs: int, rebuild_frequency: int, scheduler_patience: int, min_improvement: float):
        self.contrastive_encoder.train()
        best_overall_loss = float('inf')
        patience_counter = 0

        # 确保优化器中的 initial_lr 已设置
        for grp in self.optimizer.param_groups:
            if 'initial_lr' not in grp:
                grp['initial_lr'] = grp['lr']


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=scheduler_patience, threshold=min_improvement
        )
        self._build_and_log_datasets() # 初始数据集构建

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            if epoch > 0 and rebuild_frequency > 0 and epoch % rebuild_frequency == 0:
                print(f"Epoch {epoch+1}: 重建数据集...")
                self._build_and_log_datasets()

            self.contrastive_encoder.to(self.device) # 确保模型在正确的设备上
            self.contrastive_encoder.train() # 设置为训练模式

            epoch_losses_ds1 = []
            epoch_losses_ds2 = []

            # Dataset 1 训练循环
            if self.train_loader1 and len(self.train_loader1) > 0:
                print(f"在 Dataset1 上训练 (大小: {len(self.dataset1)} 样本, {len(self.train_loader1)} 批次)")
                progress_bar_ds1 = tqdm(self.train_loader1, desc=f"Epoch {epoch+1} DS1", leave=False, dynamic_ncols=True)
                for batch1 in progress_bar_ds1:
                    loss1_val = self._process_batch(batch1, self.loss_fn1, 'dataset1')
                    if loss1_val is not None:
                        epoch_losses_ds1.append(loss1_val.item())
                        progress_bar_ds1.set_postfix(loss=f"{loss1_val.item():.4f}")
            else:
                print("Dataset1 为空或未加载，跳过训练。")


            # Dataset 2 训练循环
            if self.train_loader2 and len(self.train_loader2) > 0:
                print(f"在 Dataset2 上训练 (大小: {len(self.dataset2)} 样本, {len(self.train_loader2)} 批次)")
                progress_bar_ds2 = tqdm(self.train_loader2, desc=f"Epoch {epoch+1} DS2", leave=False, dynamic_ncols=True)
                for batch2 in progress_bar_ds2:
                    loss2_val = self._process_batch(batch2, self.loss_fn2, 'dataset2')
                    if loss2_val is not None:
                        epoch_losses_ds2.append(loss2_val.item())
                        progress_bar_ds2.set_postfix(loss=f"{loss2_val.item():.4f}")
            else:
                print("Dataset2 为空或未加载，跳过训练。")


            avg_loss_ds1 = np.mean(epoch_losses_ds1) if epoch_losses_ds1 else 0.0
            avg_loss_ds2 = np.mean(epoch_losses_ds2) if epoch_losses_ds2 else 0.0
            current_epoch_combined_loss = 0.0
            w1, w2 = 0.0, 0.0 # 初始化权重

            if self.use_weighted_loss:
                w1_config = self.loss_weights.get('dataset1', 0.5)
                w2_config = self.loss_weights.get('dataset2', 0.5)

                active_datasets = 0
                if epoch_losses_ds1: active_datasets +=1
                if epoch_losses_ds2: active_datasets +=1

                if active_datasets == 0:
                    current_epoch_combined_loss = 0.0
                    w1, w2 = 0.0, 0.0
                elif active_datasets == 1:
                    if epoch_losses_ds1:
                        current_epoch_combined_loss = avg_loss_ds1
                        w1, w2 = 1.0, 0.0
                    else: # epoch_losses_ds2 must be active
                        current_epoch_combined_loss = avg_loss_ds2
                        w1, w2 = 0.0, 1.0
                else: # Both active
                    # Normalize configured weights
                    sum_config_w = w1_config + w2_config
                    if sum_config_w > 0:
                        w1 = w1_config / sum_config_w
                        w2 = w2_config / sum_config_w
                    else:
                        w1, w2 = 0.5, 0.5
                    current_epoch_combined_loss = (w1 * avg_loss_ds1) + (w2 * avg_loss_ds2)
            else: # 不使用加权损失，简单平均或取单个
                if epoch_losses_ds1 and epoch_losses_ds2:
                    current_epoch_combined_loss = (avg_loss_ds1 + avg_loss_ds2) / 2.0
                elif epoch_losses_ds1:
                    current_epoch_combined_loss = avg_loss_ds1
                elif epoch_losses_ds2:
                    current_epoch_combined_loss = avg_loss_ds2
                else:
                    current_epoch_combined_loss = 0.0 # 两个数据集都没有损失

            print(f"Epoch {epoch+1} 总结: 平均 DS1 损失: {avg_loss_ds1:.4f}, 平均 DS2 损失: {avg_loss_ds2:.4f}, 组合损失 (用于调度器): {current_epoch_combined_loss:.4f}")

            self.training_history['epoch'].append(epoch + 1)
            self.training_history['loss_ds1'].append(avg_loss_ds1)
            self.training_history['loss_ds2'].append(avg_loss_ds2)
            self.training_history['combined_loss'].append(current_epoch_combined_loss)
            if self.use_weighted_loss:
                self.training_history['weight_ds1'].append(w1)
                self.training_history['weight_ds2'].append(w2)

            if self.adaptive_weighting and epoch_losses_ds1 and epoch_losses_ds2 and self.use_weighted_loss:
                # 简单的基于损失的自适应权重调整 (更高级的方法存在)
                # 损失越大，权重应该越大 (假设任务是最小化损失)
                # 或者，如果任务是最大化某个指标，则指标越小，权重越大
                # 这里我们最小化损失，所以损失大的数据集应该获得更多关注（即更高的权重）
                loss_sum_for_weighting = avg_loss_ds1 + avg_loss_ds2
                if loss_sum_for_weighting > 1e-9: # 避免除以零
                    # 权重与损失成正比
                    raw_w1_adaptive = avg_loss_ds1 / loss_sum_for_weighting if avg_loss_ds1 > 0 else 0
                    raw_w2_adaptive = avg_loss_ds2 / loss_sum_for_weighting if avg_loss_ds2 > 0 else 0

                    # 平滑更新，alpha_smooth 决定新计算的权重占多大比重
                    alpha_smooth = 0.3 # 0 表示完全使用初始权重, 1 表示完全使用当前计算的权重
                    self.loss_weights['dataset1'] = (1 - alpha_smooth) * self.initial_loss_weights['dataset1'] + alpha_smooth * raw_w1_adaptive

                    self.loss_weights['dataset2'] = (1 - alpha_smooth) * self.initial_loss_weights['dataset2'] + alpha_smooth * raw_w2_adaptive

                    # 重新归一化
                    current_sum_adapted_weights = self.loss_weights['dataset1'] + self.loss_weights['dataset2']
                    if current_sum_adapted_weights > 0:
                        self.loss_weights['dataset1'] /= current_sum_adapted_weights
                        self.loss_weights['dataset2'] /= current_sum_adapted_weights
                    else: # 如果两个权重都变为零，则回退
                        self.loss_weights['dataset1'], self.loss_weights['dataset2'] = self.initial_loss_weights['dataset1'], self.initial_loss_weights['dataset2']
                    print(f"自适应权重更新: DS1={self.loss_weights['dataset1']:.3f}, DS2={self.loss_weights['dataset2']:.3f}")


            scheduler.step(current_epoch_combined_loss)
            current_lr_base = self.optimizer.param_groups[0]['lr']
            current_lr_proj = self.optimizer.param_groups[1]['lr']
            self.training_history['learning_rate_base'].append(current_lr_base)
            self.training_history['learning_rate_proj'].append(current_lr_proj)
            print(f"当前学习率: Base={current_lr_base:.2e}, Projection={current_lr_proj:.2e}")


            if current_epoch_combined_loss < best_overall_loss - min_improvement:
                if epoch_losses_ds1 or epoch_losses_ds2: # 仅当实际发生训练时才保存
                    best_overall_loss = current_epoch_combined_loss
                    patience_counter = 0
                    print(f"🎉 发现新的最佳模型! 组合损失: {best_overall_loss:.4f}. 保存模型...")
                    self.save_checkpoint(epoch + 1, best_overall_loss, is_best=True)
                else:
                    print("此轮未执行训练步骤。跳过最佳模型检查。")
            else:
                patience_counter += 1
                print(f"耐心计数: {patience_counter}/{scheduler.patience}")

            if patience_counter > scheduler.patience: # 注意：scheduler.patience 是 ReduceLROnPlateau 的参数
                print("🛑 早停触发。")
                break
            self.plot_training_progress(save_plot=False, show_plot=False) # 生成绘图数据以备保存

        print("🏁 训练完成。")
        # 保存最终模型，无论是否最佳
        # self.save_checkpoint(epoch + 1, current_epoch_combined_loss, is_best=False, final_save=True)
        self.plot_training_progress(save_plot=True, show_plot=True) # 保存并显示最终绘图


    def save_checkpoint(self, epoch_num, loss_val, is_best=False, final_save=False):
        # 只在找到更优模型时才执行保存操作
        if not is_best:
            return

        state = {
            'epoch': epoch_num,
            'contrastive_encoder_state_dict': self.contrastive_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': loss_val,
            'training_history': dict(self.training_history), # 保存训练历史的副本
            'training_model_type': self.training_model_type,
            'training_model_identifier_or_path': self.training_model_identifier_or_path,
            'projection_head_config': self.projection_config,
            'pruning_model_path': self.pruning_model_path,
            'similarity_threshold': self.similarity_threshold,
            'num_negatives': self.num_negatives,
            'batch_size': self.batch_size,
            'pruning_inference_batch_size': self.pruning_inference_batch_size,
            'base_lr_initial': self.optimizer.param_groups[0].get('initial_lr'),
            'projection_lr_initial': self.optimizer.param_groups[1].get('initial_lr'),
            'use_weighted_loss': self.use_weighted_loss,
            'loss_weights': self.loss_weights,
            'adaptive_weighting': self.adaptive_weighting,
            'infonce_mode': self.infonce_mode,
            # --- 新增PEFT状态 ---
            'use_peft': self.use_peft,
            'peft_config': self.peft_config,
            # --------------------
            'min_subtree_size_ds1': self.min_subtree_size_ds1,
            'max_samples_per_post_ds1': self.max_samples_per_post_ds1,
            'min_subtree_size_ds2': self.min_subtree_size_ds2,
            'max_samples_per_subtree_ds2': self.max_samples_per_subtree_ds2
        }
        if self.training_model_type == 'textcnn':
            state['textcnn_config'] = self.textcnn_config
            state['vocab'] = self.vocab

        # 根据模型标识符创建保存目录
        sanitized_model_name = self.training_model_identifier_or_path.replace('/', '_')
        save_dir = os.path.join("model", sanitized_model_name)
        os.makedirs(save_dir, exist_ok=True)

        # 生成损失图的PNG字节
        fig_bytes = self.plot_training_progress(save_plot=False, show_plot=False, return_bytes=True)
        if fig_bytes:
            # 将损失图字节保存到 state 中（可选，但保留了原有逻辑）
            state['loss_plot_png'] = fig_bytes
            
            # 将损失图保存为独立的PNG文件
            plot_filepath = os.path.join(save_dir, "training_loss_plot.png")
            with open(plot_filepath, 'wb') as f:
                f.write(fig_bytes)
            print(f"📊 训练损失图已保存至: {plot_filepath}")

        # 定义并保存最优模型的checkpoint文件
        filepath = os.path.join(save_dir, "best_contrastive_model.pth")
        torch.save(state, filepath)
        print(f"✅ 最优模型已更新并保存至: {filepath}")


    def plot_training_progress(self, save_plot=False, show_plot=True, return_bytes=False):
        if not self.training_history['epoch']:
            if return_bytes: return None
            return

        try:
            import matplotlib
            matplotlib.use('Agg') # 确保非交互式后端
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-v0_8-whitegrid')
        except ImportError:
            print("Matplotlib 未找到。跳过绘图生成。")
            if return_bytes: return None
            return
        except UserWarning: pass # 忽略seaborn样式警告

        fig, ax1 = plt.subplots(figsize=(14, 8)) # 调整图形大小
        epochs_data = self.training_history['epoch']

        color_ds1_loss = 'orangered'
        color_ds2_loss = 'forestgreen'
        color_combined_loss = 'mediumblue'

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', color='black', fontsize=12)

        if 'loss_ds1' in self.training_history and any(v is not None and v > 0 for v in self.training_history['loss_ds1']):
            ax1.plot(epochs_data, self.training_history['loss_ds1'], color=color_ds1_loss, linestyle='--', marker='.', markersize=6, label='DS1 Loss (父子)')
        if 'loss_ds2' in self.training_history and any(v is not None and v > 0 for v in self.training_history['loss_ds2']):
            ax1.plot(epochs_data, self.training_history['loss_ds2'], color=color_ds2_loss, linestyle=':', marker='x', markersize=6, label='DS2 Loss (节点-中心)')
        if 'combined_loss' in self.training_history and any(v is not None for v in self.training_history['combined_loss']):
            ax1.plot(epochs_data, self.training_history['combined_loss'], color=color_combined_loss, linewidth=2.5, marker='o', markersize=4, label='组合损失 (调度器)')
        ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.set_yscale('log') # 尝试对数刻度，如果损失范围很大

        ax2 = ax1.twinx()
        color_lr_base = 'purple'
        color_lr_proj = 'darkcyan'
        color_w1 = 'coral'
        color_w2 = 'lightgreen'

        ax2.set_ylabel('学习率 / 权重', color='dimgray', fontsize=12)
        if 'learning_rate_base' in self.training_history and any(v is not None for v in self.training_history['learning_rate_base']):
            ax2.plot(epochs_data, self.training_history['learning_rate_base'], color=color_lr_base, linestyle='-.', marker='s', markersize=4, label='LR (Base)')
        if 'learning_rate_proj' in self.training_history and any(v is not None for v in self.training_history['learning_rate_proj']):
            ax2.plot(epochs_data, self.training_history['learning_rate_proj'], color=color_lr_proj, linestyle='-.', marker='D', markersize=3, label='LR (Proj)')

        if self.use_weighted_loss and 'weight_ds1' in self.training_history and 'weight_ds2' in self.training_history:
            if any(v is not None for v in self.training_history['weight_ds1']):
                 ax2.plot(epochs_data, self.training_history['weight_ds1'], color=color_w1, linestyle=(0, (3, 5, 1, 5)), marker='^', markersize=5, label='权重 DS1') # (0, (3, 5, 1, 5)) is dashdotdotted
            if any(v is not None for v in self.training_history['weight_ds2']):
                 ax2.plot(epochs_data, self.training_history['weight_ds2'], color=color_w2, linestyle=(0, (3, 5, 1, 5)), marker='v', markersize=5, label='权重 DS2')

        ax2.tick_params(axis='y', labelcolor='dimgray', labelsize=10)
        ax2.set_ylim(bottom=0) # 学习率和权重非负
        # ax2.set_yscale('log') # 如果学习率变化范围也很大

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        # 将图例放在图表下方
        fig.legend(handles1 + handles2, labels1 + labels2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, fontsize=9)

        model_name_for_title = self.training_model_identifier_or_path.split('/')[-1]
        plt.title(f'训练进度: {model_name_for_title} ({self.training_model_type.upper()})', fontsize=15, pad=25)
        fig.tight_layout(rect=[0, 0.08, 1, 0.95]) # 调整布局为图例留出空间

        plot_bytes_val = None
        if return_bytes or save_plot:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight') # bbox_inches='tight' 确保所有内容都保存
            buf.seek(0)
            plot_bytes_val = buf.getvalue()
            buf.close()

        # if save_plot and plot_bytes_val:
        #     plot_filename = f"training_progress_{self.training_model_identifier_or_path.replace('/', '_')}_{self.training_model_type}.png"
        #     with open(plot_filename, 'wb') as f:
        #         f.write(plot_bytes_val)
        #     print(f"📈 训练进度图已保存到 {plot_filename}")

        # if show_plot:
        #     # 由于使用了 'Agg' 后端，plt.show() 不会显示任何内容。
        #     # 用户需要打开保存的PNG文件来查看图像。
        #     print("绘图已生成。如果 save_plot=True，请检查保存的PNG文件。")

        plt.close(fig) # 关闭图形以释放内存

        if return_bytes:
            return plot_bytes_val
        return None

def build_pruned_forest(post_storage: PostStorage, similarity_threshold: float):
    """
    基于相似度阈值构建剪枝后的森林
    """
    print("🔄 构建剪枝森林...")
    post_storage.forests.clear()
    pruning_results = post_storage.prune_all_posts_by_similarity(
        similarity_threshold=similarity_threshold, show_progress=True
    )
    print(f"✅ 森林构建完成: {len(pruning_results)} 个帖子")
    return pruning_results

def fine_tune_contrastive_model(
    model: ContrastiveEncoder,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ContrastiveLoss,
    device: torch.device,
    num_epochs: int = 3,
    scheduler_patience: int = 2,
    min_improvement: float = 1e-5
):
    """
    对比模型微调
    """
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, threshold=min_improvement
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batches_processed = 0

        progress_bar = tqdm(train_loader, desc=f"微调对比模型 (Epoch {epoch+1}/{num_epochs})", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            
            anchor_texts = batch['anchor_texts']
            positive_texts_ds1 = batch['positive_texts_ds1']
            negative_texts = batch['negative_texts']
            num_negatives = batch['num_negatives']

            # 过滤掉 positive_texts_ds1 中的 None (来自Dataset2的占位符)
            valid_indices_ds1 = [i for i, txt in enumerate(positive_texts_ds1) if txt is not None]
            
            if valid_indices_ds1:
                anchor_texts_ds1 = [anchor_texts[i] for i in valid_indices_ds1]
                positive_texts_ds1_f = [positive_texts_ds1[i] for i in valid_indices_ds1]

                if anchor_texts_ds1: # 确保列表不为空
                    anchor_emb = model(anchor_texts_ds1)
                    positive_emb_ds1 = model(positive_texts_ds1_f)
                    
                    # 重构与过滤后样本对应的负样本
                    neg_emb = None
                    if negative_texts and num_negatives > 0:
                        # 从扁平列表中提取与 valid_indices_ds1 对应的负样本
                        current_batch_neg_texts = []
                        original_batch_size_collator = len(anchor_texts) # collator处理前的batch大小
                        for orig_idx in valid_indices_ds1:
                            start = orig_idx * num_negatives
                            end = start + num_negatives
                            current_batch_neg_texts.extend(negative_texts[start:end])
                        
                        if current_batch_neg_texts:
                            neg_emb_flat = model(current_batch_neg_texts)
                            neg_emb = neg_emb_flat.view(len(anchor_texts_ds1), num_negatives, -1)
                    
                    if neg_emb is not None and neg_emb.nelement() > 0:
                        loss = loss_fn(anchor_emb, positive_emb_ds1, neg_emb)
                    else:
                        loss = loss_fn(anchor_emb, positive_emb_ds1)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    batches_processed += 1

            progress_bar.set_postfix(loss=epoch_loss / batches_processed if batches_processed > 0 else epoch_loss)

        avg_loss = epoch_loss / batches_processed if batches_processed > 0 else 0.0
        scheduler.step(avg_loss)

        if avg_loss < best_loss - min_improvement:
            best_loss = avg_loss
            patience_counter = 0
            print(f"💾 保存当前最佳模型，损失: {best_loss:.4f}")
            # 可以选择保存模型
            # torch.save(model.state_dict(), "best_contrastive_model.pth")
        else:
            patience_counter += 1
            print(f"⏳ 等待更优模型，当前计数器: {patience_counter}/{scheduler_patience}")

        if patience_counter >= scheduler_patience:
            print("🛑 早停触发，停止训练")
            break

def main_jina_training_pipeline():
    print("🚀 开始训练流程...")
    # 1. 准备数据
    # try:
    #     with open('comments_data.json', 'r', encoding='utf-8') as f:
    #         comment_data = json.load(f)
    #     with open('contents_data.json', 'r', encoding='utf-8') as f:
    #         post_data = json.load(f)
    # except FileNotFoundError:
    #     print("错误: comments_data.json 或 contents_data.json 未找到。请确保数据文件存在。")
    #     return

    # comment_df = pd.DataFrame(comment_data)
    # post_df = pd.DataFrame(post_data)

    # # 采样评论数量最少的两个note_id的数据
    # # 统计每个note_id的评论数量
    # note_id_counts = comment_df['note_id'].value_counts()

    # # 获取评论数量最少的note_id
    # # 如果note_id数量少于2，则取所有
    # if len(note_id_counts) > 0:
    #     num_to_sample = min(2, len(note_id_counts))
    #     sampled_note_ids = note_id_counts.nsmallest(num_to_sample).index.tolist()
    #     print(f"采样评论数量最少的 {num_to_sample} 个 note_id: {sampled_note_ids}")

    #     # 根据选中的note_id筛选数据
    #     comment_df = comment_df[comment_df['note_id'].isin(sampled_note_ids)]
    #     post_df = post_df[post_df['note_id'].isin(sampled_note_ids)]
    #     print(f"采样后，comment_df 形状: {comment_df.shape}, post_df 形状: {post_df.shape}")
    # else:
    #     print("警告: comment_df 中没有 note_id 可供采样。使用所有可用数据。")

    # 1. 准备数据
    try:
        comment_df = pd.read_csv('data_process/cl_data/train_comments_filtered.csv', encoding='utf-8')
        post_df = pd.read_csv('data_process/cl_data/train_posts_filtered.csv', encoding='utf-8')
    except FileNotFoundError:
        print("错误: comments_data.csv 或 contents_data.csv 未找到。请确保数据文件存在。")
        return      
    

    # 确保 note_id 和 comment_id 是字符串类型，以避免后续问题
    comment_df['note_id'] = comment_df['note_id'].astype(str)
    comment_df['comment_id'] = comment_df['comment_id'].astype(str)
    comment_df['parent_comment_id'] = comment_df['parent_comment_id'].astype(str)
    post_df['note_id'] = post_df['note_id'].astype(str)


    storage = PostStorage()
    # 确保帖子内容是字符串，如果 title 不存在，尝试 content，如果都为空，则为空字符串
    for _, row in post_df.iterrows():
        post_content = str(row.get('title', '')) or str(row.get('content', '')) # 保证是字符串
        storage.add_post(post_id=str(row['note_id']), post_content=post_content)

    for _, row in comment_df.iterrows():
        post_id_str = str(row['note_id'])
        comment_id_str = str(row['comment_id'])
        content_str = str(row.get('content', '')) # 确保内容是字符串
        parent_id_str = str(row['parent_comment_id']) if str(row['parent_comment_id']) != '0' else post_id_str
        
        try:
            storage.add_comment_to_post(post_id_str, comment_id_str, content_str, parent_id_str)
        except Exception as e:
            print(f"插入评论失败: {e}, 帖子ID: {post_id_str}, 评论ID: {comment_id_str}")

    # 2. 选择训练模型类型并配置训练器
    common_trainer_params = {
        'post_storage': storage,
        'pruning_model_path': "google-bert/bert-base-chinese", # 
        'similarity_threshold': 0.7, # 调整阈值
        'num_negatives': 8,      # 增加负样本数量
        'batch_size': 8,        # 调整批量大小
        'pruning_inference_batch_size': 16, # <--- 为剪枝模型推断设置一个合理的批大小
        'base_lr': 5e-6,         # 调整学习率
        'projection_lr': 5e-5,
        'use_weighted_loss': True,
        'loss_weights': {'dataset1': 1, 'dataset2': 0}, # 调整权重
        'adaptive_weighting': False, # 启用自适应权重
        'infonce_mode': 'bidirectional', # 双向对比
        'projection_head_config': {'hidden_dim': 768, 'output_dim': 384, 'dropout_rate': 0.15}, # 调整投影头
        'min_subtree_size_ds1': 2, 'max_samples_per_post_ds1': None,
        'min_subtree_size_ds2': 100000, 'max_samples_per_subtree_ds2': None,

        # --- 新增PEFT配置 ---
        'use_peft': False,  # 设置为 True 来启用 LoRA
        'peft_config': {
            'r': 8,              # LoRA的秩，越小参数越少，常用8, 16, 32
            'lora_alpha': 16,    # LoRA的缩放因子，通常是r的两倍
            'target_modules': ["query", "key", "value"], # 对注意力的Q,K,V应用
            'lora_dropout': 0.1, # LoRA层的dropout率
            'bias': "none",      # "none", "all", "lora_only"
    }}

    # # 🎯 选项 1: ModelScope 模型
    # print("\n--- 配置 ModelScope 模型训练 ---")
    # trainer = DynamicContrastiveTrainer(
    #     training_model_type='modelscope',
    #     # 使用另一个ModelScope模型作为训练目标
    #     training_model_identifier_or_path="google-bert/bert-base-chinese",
    #     **common_trainer_params
    # )

    # 🎯 选项 2: 自定义 TextCNN
    print("\n--- 配置 TextCNN 训练 ---")
    textcnn_specific_config = {
        'embedding_dim': 300,       
        'num_filters': 128,         
        'filter_sizes': [2, 3, 4],  
        'model_dropout_rate': 0.1,  
        'max_seq_length': 200,      # TextCNN分词器的最大序列长度
        'textcnn_output_dim': 768,  # TextCNN输出维度 (与投影头输出匹配或作为其输入)
        'min_vocab_freq': 1         # 词汇表最小词频
    }
    # 确保 TextCNN 的输出维度与投影头的输入维度匹配
    # common_trainer_params['projection_head_config']['hidden_dim'] 可以基于 textcnn_output_dim
    # 或者 textcnn_output_dim 直接作为投影头的输入
    # 这里假设 textcnn_output_dim 是投影头的输入，所以 base_dim 会是 textcnn_output_dim

    trainer = DynamicContrastiveTrainer(
        training_model_type='textcnn',
        training_model_identifier_or_path="model/my_custom_textcnn_v4_no_pruning_paircl", # 自定义模型标识符
        textcnn_config=textcnn_specific_config,
        **common_trainer_params
    )
    
    # 3. 开始训练
    print("\n--- 开始训练 ---")
    trainer.train(
        num_epochs=2, # 为了快速测试，减少了epoch，原为100
        rebuild_frequency=2,  # 为了快速测试，减少了频率，原为200
        scheduler_patience=7, # 原为2
        min_improvement=1e-5
    )
    
    print("🎉 训练流程完成!")
    print(f"💾 最佳模型和训练状态已保存。训练后的基础模型部分位于 'trained_{trainer.training_model_type}_embedding_model' 目录中。")

def load_trained_model_and_tokenizer(checkpoint_path: str):
    """
    从checkpoint加载完整的训练器状态，并返回训练好的基础模型和分词器。
    """
    if not os.path.exists(checkpoint_path):
        print(f"错误: Checkpoint 文件 {checkpoint_path} 未找到。")
        return None, None, None

    print(f"正在从 {checkpoint_path} 加载checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'),weights_only=False) # 加载到CPU以避免GPU问题

    model_type = checkpoint['training_model_type']
    model_identifier = checkpoint['training_model_identifier_or_path']
    proj_config = checkpoint['projection_head_config']

    # --- 新增：获取PEFT配置 ---
    use_peft = checkpoint.get('use_peft', False)
    peft_config = checkpoint.get('peft_config', None)
    # -------------------------
    
    encoder = None
    if model_type == 'textcnn':
        textcnn_conf = checkpoint['textcnn_config']
        vocab_data = checkpoint['vocab']
        encoder = ContrastiveEncoder(
            model_type='textcnn',
            vocab=vocab_data,
            textcnn_config=textcnn_conf,
            projection_hidden_dim=proj_config['hidden_dim'],
            projection_output_dim=proj_config['output_dim'],
            projection_dropout_rate=proj_config['dropout_rate']
        )
    elif model_type == 'modelscope':
        encoder = ContrastiveEncoder(
            model_type='modelscope',
            model_name_or_path=model_identifier,
            projection_hidden_dim=proj_config['hidden_dim'],
            projection_output_dim=proj_config['output_dim'],
            projection_dropout_rate=proj_config['dropout_rate']
        )
        # --- 新增：如果使用了PEFT，重新包装模型 ---
        if use_peft:
            print("🔧 检测到PEFT训练的checkpoint，正在重新应用LoRA配置...")
            lora_config = LoraConfig(**peft_config)
            encoder.base_model = get_peft_model(encoder.base_model, lora_config)
            print("✅ LoRA配置已重新应用。")
        # ------------------------------------
    else:
        print(f"错误: Checkpoint中未知的模型类型 '{model_type}'")
        return None, None, None

    encoder.load_state_dict(checkpoint['contrastive_encoder_state_dict'])
    encoder.eval() # 设置为评估模式
    print(f"✅ {model_type.upper()} ContrastiveEncoder 加载完成并设置为评估模式。")
    
    # 返回基础模型和分词器
    return encoder.base_model, encoder.tokenizer, model_type


if __name__ == "__main__":
    # # 确保jieba已初始化（如果它是惰性加载的）
    # try:
    #     _ = jieba.lcut("测试jieba初始化")
    # except Exception as e:
    #     print(f"由于以下原因初始化jieba: {e}")
    #     jieba.initialize() # 显式初始化

    main_jina_training_pipeline()






