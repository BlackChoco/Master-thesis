import torch
from torch.utils.data import Dataset
import numpy as np
import random
import re
import jieba
from typing import List, Dict, Tuple, Optional, Union
from Tree_data_model import PostStorage
from collections import defaultdict, Counter
from tqdm import tqdm


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
    processed_text = re.sub(r'@\S+\s?', '', processed_text)

    # 2. 去掉厂商的名字
    brand_names = [
        '小米', '苹果', '三星', '荣耀', '华为', '一加', 
        'oppo', 'OPPO', 'vivo', 'realme', '红米', '真我','安卓',
        'x7pro','x100s','gt5pro','GTneo5','pro','手机','11','12','13','14','备用机',
        'p40'
    ]
    brand_pattern = re.compile('|'.join(brand_names), re.IGNORECASE)
    processed_text = brand_pattern.sub('', processed_text)
    
    return processed_text.strip()


def build_vocab_from_post_storage(post_storage: PostStorage, min_freq: int = 1) -> Tuple[Dict[str, int], int]:
    """
    使用jieba从PostStorage中的所有评论内容构建词汇表。
    """
    print("正在使用 jieba 构建词汇表...")
    word_counts = Counter()
    
    all_comments_content = []
    for post_id, post_tree in post_storage.posts.items():
        def collect_content_from_node(node):
            if node.content and isinstance(node.content, str):
                all_comments_content.append(node.content)
            else:
                pass
            for child in node.children:
                collect_content_from_node(child)
        if post_tree.root:
            collect_content_from_node(post_tree.root)

    print(f"找到 {len(all_comments_content)} 条评论用于构建词汇表。")
    
    for text in tqdm(all_comments_content, desc="评论分词中"):
        if text and isinstance(text, str):
            try:
                seg_list = jieba.lcut(text.strip())
                word_counts.update(seg_list)
            except Exception as e:
                print(f"警告: jieba 分词失败，文本: '{text[:50]}...'，错误: {e}")
        elif not text:
            pass
        else:
            print(f"警告: 无效的文本类型进行分词: {type(text)}, 内容: {text[:50]}...")

    # 创建词汇表
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    print(f"词汇表构建完成，包含 {len(vocab)} 个独立词元 (min_freq={min_freq})。")
    return vocab, len(vocab)


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
            return default_str
        return content

    def _build_dataset(self, max_samples_per_post: Optional[int]):
        print("🔨 构建Dataset1: 父子评论相似度对比学习数据集")
        total_pairs = 0
        for post_id, forest in tqdm(self.post_storage.forests.items(), desc="处理帖子(Dataset1)"):
            post_pairs = []
            if forest.subtrees is None:
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

                parent_content_clean = preprocess_text(node_content_raw)
                child_content_clean = preprocess_text(child_content_raw)

                if len(parent_content_clean) < 5 or len(child_content_clean) < 5:
                    traverse_node(child)
                    continue

                try:
                    similarity = node.calculate_similarity(child)
                    if similarity >= self.similarity_threshold:
                        pairs.append({
                            'parent_content': parent_content_clean,
                            'child_content': child_content_clean,
                            'parent_id': node.comment_id,
                            'child_id': child.comment_id,
                            'similarity': similarity,
                            'post_id': post_id
                        })
                except (ValueError, AttributeError, TypeError) as e:
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
                    if subtree_node_contents:
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
            content_clean = preprocess_text(content_raw)
            if len(content_clean) >= 5:
                contents.append(content_clean)
            for child in node.children:
                collect_contents(child)
        if root: collect_contents(root)
        return contents

    def _extract_node_center_pairs(self, root, subtree_node_contents: List[str], post_id,
                                 max_samples: Optional[int]) -> List[Dict]:
        pairs = []
        if not subtree_node_contents: return []

        all_nodes_in_subtree = []
        def _collect_all_nodes(node):
            all_nodes_in_subtree.append(node)
            for child in node.children:
                _collect_all_nodes(child)
        if root: _collect_all_nodes(root)
        
        for node in all_nodes_in_subtree:
            node_content_raw = self._ensure_str_content(node.content)
            node_content_clean = preprocess_text(node_content_raw)
            if len(node_content_clean) < 5:
                continue

            if node_content_clean: 
                pairs.append({
                    'node_content': node_content_clean,
                    'center_node_contents': subtree_node_contents,
                    'node_id': node.comment_id,
                    'post_id': post_id,
                    'subtree_size': len(all_nodes_in_subtree)
                })
        
        if max_samples and len(pairs) > max_samples:
            pairs = random.sample(pairs, max_samples)
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
            'anchor_content': self._ensure_str_content(pair['node_content']),
            'positive_content_list': [self._ensure_str_content(c) for c in pair['center_node_contents']],
            'post_id': pair['post_id'],
            'subtree_size': pair['subtree_size'],
            'pair_type': 'node_center',
            'is_center_embedding': False
        }

    def get_negative_samples(self, post_id: str, num_negatives: int = 1) -> List[str]:
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


class ContrastiveDataCollator:
    """
    自定义的DataCollator，用于批量处理并动态添加负样本
    """
    def __init__(self, dataset: Union[ContrastiveDataset1, ContrastiveDataset2], num_negatives: int = 2):
        self.dataset = dataset
        self.num_negatives = num_negatives
    
    def _ensure_str_content(self, content, default_str="<unk>"):
        if not isinstance(content, str):
            return default_str
        return content if content else default_str

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
                content_list = item.get('positive_content_list', [])
                if isinstance(content_list, list):
                    processed_list = [self._ensure_str_content(c) for c in content_list]
                    positive_content_lists_ds2.append(processed_list if processed_list else [self._ensure_str_content("")])
                else:
                    positive_content_lists_ds2.append([self._ensure_str_content(content_list)])
            else:
                positive_texts_ds1.append(self._ensure_str_content(item.get('positive_content', item['anchor_content'])))
                positive_content_lists_ds2.append(None)
        
        negative_texts = []
        if self.num_negatives > 0:
            for item in batch:
                post_id = item['post_id']
                neg_contents = []
                if hasattr(self.dataset, 'get_negative_samples'):
                    neg_contents = self.dataset.get_negative_samples(post_id, self.num_negatives)
                
                processed_neg_contents = []
                for neg_c in neg_contents:
                    processed_neg_contents.append(self._ensure_str_content(neg_c))
                
                while len(processed_neg_contents) < self.num_negatives:
                    other_items_texts = [self._ensure_str_content(b['anchor_content']) for b in batch if b['post_id'] != post_id and self._ensure_str_content(b['anchor_content'])]
                    if other_items_texts:
                        processed_neg_contents.append(random.choice(other_items_texts))
                    else:
                        processed_neg_contents.append(self._ensure_str_content("")) 
                
                negative_texts.extend(processed_neg_contents[:self.num_negatives])
        
        return {
            'anchor_texts': anchor_texts,
            'positive_texts_ds1': positive_texts_ds1, 
            'positive_content_lists_ds2': positive_content_lists_ds2, 
            'negative_texts': negative_texts,
            'post_ids': [item['post_id'] for item in batch],
            'pair_types': [item.get('pair_type', 'unknown') for item in batch],
            'num_negatives': self.num_negatives
        }