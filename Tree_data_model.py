import numpy as np
from typing import List, Callable, Optional, Dict, Tuple
import copy
import torch
from tqdm import tqdm
import random

class CommentNode:
    """
    表示一条评论或回复的节点。
    """
    def __init__(self, comment_id, content, parent_id=None):
        self.comment_id = comment_id      # 评论唯一ID
        self.content = content            # 评论内容
        self.parent_id = parent_id        # 父评论ID（根评论为None或帖子ID）
        self.children = []                # 子评论列表
        self.embedding = None             # 存储节点的向量表示
        self.similarity_scores = {}       # 存储与其他节点的相似度分数

    def add_child(self, child_node):
        """
        添加一个子评论节点。
        """
        self.children.append(child_node)
    
    def set_embedding(self, embedding):
        """
        设置节点的向量表示
        """
        self.embedding = embedding
    
    def calculate_similarity(self, other_node, similarity_func=None):
        """
        计算与另一个节点的相似度
        
        参数:
            other_node: 另一个CommentNode对象
            similarity_func: 自定义相似度计算函数，默认使用余弦相似度
        
        返回:
            相似度分数(0-1之间)
        """
        if self.embedding is None or other_node.embedding is None:
            raise ValueError("节点缺少向量表示，无法计算相似度")
        
        if similarity_func is None:
            # 默认使用余弦相似度
            return self._cosine_similarity(self.embedding, other_node.embedding)
        else:
            return similarity_func(self.embedding, other_node.embedding)
    
    def _cosine_similarity(self, vec1, vec2):
        """
        计算两个向量的余弦相似度
        """
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        return np.dot(vec1_norm, vec2_norm)
    
    def deep_copy(self):
        """
        深度复制节点及其所有子节点
        """
        copied_node = CommentNode(self.comment_id, self.content, self.parent_id)
        copied_node.embedding = self.embedding.copy() if self.embedding is not None else None
        copied_node.similarity_scores = self.similarity_scores.copy()
        
        for child in self.children:
            copied_child = child.deep_copy()
            copied_node.add_child(copied_child)
        
        return copied_node

class ForestManager:
    """
    管理剪边后产生的多棵子树的森林
    """
    def __init__(self, original_post_tree=None):
        self.original_post_tree = original_post_tree  # 原始帖子树
        self.subtrees = []  # 存储所有子树（一级评论及其子树）
        self.cut_edges = []  # 存储被剪掉的边信息
        self.similarity_threshold = 0.0  # 相似度阈值
    
    def add_subtree(self, subtree_root, original_parent_id=None):
        """
        添加一棵子树到森林
        """
        subtree_info = {
            'root': subtree_root,
            'original_parent_id': original_parent_id,
            'size': self._count_nodes(subtree_root),
            'depth': self._get_tree_depth(subtree_root)
        }
        self.subtrees.append(subtree_info)
    
    def record_cut_edge(self, parent_id, child_id, similarity_score, reason="low_similarity"):
        """
        记录被剪掉的边信息
        
        参数:
            parent_id: 父节点ID
            child_id: 子节点ID  
            similarity_score: 相似度分数
            reason: 剪边原因
        """
        edge_info = {
            'parent_id': parent_id,
            'child_id': child_id,
            'similarity_score': similarity_score,
            'threshold': self.similarity_threshold,
            'reason': reason,
            'timestamp': None  # 可以添加时间戳
        }
        self.cut_edges.append(edge_info)
    
    def get_subtree_count(self):
        """获取子树数量"""
        return len(self.subtrees)
    
    def get_subtree_by_index(self, index):
        """通过索引获取子树"""
        if 0 <= index < len(self.subtrees):
            return self.subtrees[index]
        return None
    
    def get_largest_subtrees(self, n=5):
        """获取最大的n棵子树"""
        sorted_subtrees = sorted(self.subtrees, key=lambda x: x['size'], reverse=True)
        return sorted_subtrees[:n]
    
    def get_cut_edges_count(self):
        """获取被剪掉的边数量"""
        return len(self.cut_edges)
    
    def get_cut_edges_by_similarity_range(self, min_sim=0.0, max_sim=1.0):
        """获取指定相似度范围内被剪掉的边"""
        return [edge for edge in self.cut_edges 
                if min_sim <= edge['similarity_score'] <= max_sim]
    
    def get_cut_edges_statistics(self):
        """获取剪边的统计信息"""
        if not self.cut_edges:
            return {"error": "没有剪边记录"}
        
        similarities = [edge['similarity_score'] for edge in self.cut_edges 
                       if edge['similarity_score'] >= 0]  # 排除无法计算相似度的情况
        
        if not similarities:
            return {"error": "没有有效的相似度记录"}
        
        return {
            'total_cut_edges': len(self.cut_edges),
            'valid_similarity_count': len(similarities),
            'avg_similarity': np.mean(similarities),
            'min_similarity': min(similarities),
            'max_similarity': max(similarities),
            'std_similarity': np.std(similarities),
            'median_similarity': np.median(similarities)
        }
    
    def _count_nodes(self, node):
        """计算节点数量"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _get_tree_depth(self, node, depth=0):
        """计算树的最大深度"""
        if not node.children:
            return depth
        return max(self._get_tree_depth(child, depth + 1) for child in node.children)
    
    def get_forest_statistics(self):
        """获取森林的统计信息"""
        if not self.subtrees:
            return {}
        
        sizes = [subtree['size'] for subtree in self.subtrees]
        depths = [subtree['depth'] for subtree in self.subtrees]
        
        return {
            'subtree_count': len(self.subtrees),
            'total_nodes': sum(sizes),
            'avg_subtree_size': np.mean(sizes),
            'max_subtree_size': max(sizes),
            'min_subtree_size': min(sizes),
            'avg_depth': np.mean(depths),
            'max_depth': max(depths),
            'cut_edges_count': len(self.cut_edges)
        }
    
    def analyze_node_distribution(self):
        """
        分析森林中节点分布情况
        
        返回:
            包含详细节点分布信息的字典
        """
        if not self.subtrees:
            return {
                'total_nodes': 0,
                'subtree_count': 0,
                'avg_nodes_per_subtree': 0,
                'max_subtree_size': 0,
                'min_subtree_size': 0,
                'histogram': {},
                'size_distribution': {},
                'detailed_stats': {}
            }
        
        # 收集所有子树的节点数量
        tree_sizes = [subtree['size'] for subtree in self.subtrees]
        
        # 基本统计
        total_nodes = sum(tree_sizes)
        subtree_count = len(tree_sizes)
        avg_size = total_nodes / subtree_count
        max_size = max(tree_sizes)
        min_size = min(tree_sizes)
        
        # 计算直方图 - 按节点数量范围分组
        histogram_ranges = [1, 2, 3, 5, 10, 20, 50, 100]
        histogram = {}
        
        for i in range(len(histogram_ranges)):
            if i == 0:
                range_key = f"1"
                count = len([s for s in tree_sizes if s == 1])
            elif i == len(histogram_ranges) - 1:
                range_key = f"{histogram_ranges[i]}+"
                count = len([s for s in tree_sizes if s >= histogram_ranges[i]])
            else:
                prev_range = histogram_ranges[i-1]
                curr_range = histogram_ranges[i]
                range_key = f"{prev_range+1}-{curr_range}"
                count = len([s for s in tree_sizes if prev_range < s <= curr_range])
            
            histogram[range_key] = {
                'count': count,
                'percentage': (count / subtree_count * 100) if subtree_count > 0 else 0
            }
        
        # 按规模分类子树
        small_trees = len([s for s in tree_sizes if s <= 3])  # 小型：1-3个节点
        medium_trees = len([s for s in tree_sizes if 3 < s <= 10])  # 中型：4-10个节点
        large_trees = len([s for s in tree_sizes if s > 10])  # 大型：>10个节点
        
        size_distribution = {
            'small_trees': {
                'count': small_trees,
                'percentage': (small_trees / subtree_count * 100) if subtree_count > 0 else 0,
                'description': '1-3个节点'
            },
            'medium_trees': {
                'count': medium_trees,
                'percentage': (medium_trees / subtree_count * 100) if subtree_count > 0 else 0,
                'description': '4-10个节点'
            },
            'large_trees': {
                'count': large_trees,
                'percentage': (large_trees / subtree_count * 100) if subtree_count > 0 else 0,
                'description': '>10个节点'
            }
        }
        
        # 详细统计信息
        detailed_stats = {
            'median_size': np.median(tree_sizes),
            'std_size': np.std(tree_sizes),
            'percentiles': {
                '25th': np.percentile(tree_sizes, 25),
                '75th': np.percentile(tree_sizes, 75),
                '90th': np.percentile(tree_sizes, 90),
                '95th': np.percentile(tree_sizes, 95)
            },
            'size_frequency': {size: tree_sizes.count(size) for size in set(tree_sizes)}
        }
        
        return {
            'total_nodes': total_nodes,
            'subtree_count': subtree_count,
            'avg_nodes_per_subtree': avg_size,
            'max_subtree_size': max_size,
            'min_subtree_size': min_size,
            'histogram': histogram,
            'size_distribution': size_distribution,
            'detailed_stats': detailed_stats
        }
    
    def print_node_distribution_report(self):
        """
        打印格式化的节点分布报告
        """
        analysis = self.analyze_node_distribution()
        
        print("=" * 80)
        print("🌲 森林节点分布分析报告")
        print("=" * 80)
        
        # 基本统计
        print(f"📊 基本统计:")
        print(f"   总节点数: {analysis['total_nodes']}")
        print(f"   子树数量: {analysis['subtree_count']}")
        print(f"   平均每棵子树节点数: {analysis['avg_nodes_per_subtree']:.2f}")
        print(f"   最大子树节点数: {analysis['max_subtree_size']}")
        print(f"   最小子树节点数: {analysis['min_subtree_size']}")
        
        # 详细统计
        detailed = analysis['detailed_stats']
        print(f"\n📈 详细统计:")
        print(f"   中位数: {detailed['median_size']:.2f}")
        print(f"   标准差: {detailed['std_size']:.2f}")
        print(f"   25th百分位: {detailed['percentiles']['25th']:.2f}")
        print(f"   75th百分位: {detailed['percentiles']['75th']:.2f}")
        print(f"   90th百分位: {detailed['percentiles']['90th']:.2f}")
        print(f"   95th百分位: {detailed['percentiles']['95th']:.2f}")
        
        # 直方图
        print(f"\n📊 节点数量分布直方图:")
        for range_key, data in analysis['histogram'].items():
            if data['count'] > 0:
                bar = "█" * min(50, data['count'])
                print(f"   {range_key:>8} 节点 | {bar} ({data['count']} 棵, {data['percentage']:.1f}%)")
        
        # 规模分类
        print(f"\n🏷️  子树规模分类:")
        for category, data in analysis['size_distribution'].items():
            category_name = {
                'small_trees': '小型子树',
                'medium_trees': '中型子树', 
                'large_trees': '大型子树'
            }.get(category, category)
            
            print(f"   {category_name} ({data['description']}): {data['count']} 棵 ({data['percentage']:.1f}%)")
        
        # 精确频率分布（仅显示前10个最常见的大小）
        print(f"\n🔍 精确节点数频率分布 (前10名):")
        sorted_freq = sorted(detailed['size_frequency'].items(), key=lambda x: x[1], reverse=True)
        for size, freq in sorted_freq[:10]:
            percentage = (freq / analysis['subtree_count'] * 100) if analysis['subtree_count'] > 0 else 0
            print(f"   {size} 个节点: {freq} 棵子树 ({percentage:.1f}%)")
        
        if len(sorted_freq) > 10:
            print(f"   ... 还有 {len(sorted_freq) - 10} 种不同的节点数量")
        
        print("=" * 80)
    
    def get_subtrees_by_size_range(self, min_size=1, max_size=float('inf')):
        """
        获取指定大小范围内的子树
        
        参数:
            min_size: 最小节点数
            max_size: 最大节点数
        
        返回:
            符合条件的子树列表
        """
        return [subtree for subtree in self.subtrees 
                if min_size <= subtree['size'] <= max_size]
    
    def get_distribution_summary(self):
        """
        获取分布摘要信息
        
        返回:
            简化的分布摘要
        """
        analysis = self.analyze_node_distribution()
        
        return {
            'summary': f"森林包含 {analysis['subtree_count']} 棵子树，总计 {analysis['total_nodes']} 个节点",
            'avg_size': f"平均每棵子树 {analysis['avg_nodes_per_subtree']:.1f} 个节点",
            'size_range': f"子树大小范围: {analysis['min_subtree_size']}-{analysis['max_subtree_size']} 个节点",
            'dominant_category': self._get_dominant_category(analysis['size_distribution']),
            'fragmentation_level': self._calculate_fragmentation_level(analysis)
        }
    
    def _get_dominant_category(self, size_distribution):
        """获取占主导地位的子树类别"""
        max_category = max(size_distribution.items(), key=lambda x: x[1]['count'])
        category_name = {
            'small_trees': '小型子树',
            'medium_trees': '中型子树',
            'large_trees': '大型子树'
        }.get(max_category[0], max_category[0])
        
        return f"{category_name} 占主导 ({max_category[1]['percentage']:.1f}%)"
    
    def _calculate_fragmentation_level(self, analysis):
        """计算森林的碎片化程度"""
        if analysis['subtree_count'] <= 1:
            return "无碎片化"
        
        # 基于子树数量和平均大小计算碎片化程度
        avg_size = analysis['avg_nodes_per_subtree']
        subtree_count = analysis['subtree_count']
        
        if avg_size >= 10:
            return "低碎片化"
        elif avg_size >= 5:
            return "中等碎片化"
        elif avg_size >= 2:
            return "高碎片化"
        else:
            return "极高碎片化"

class PostTree:
    """
    表示一个帖子及其所有评论的树结构。
    """
    def __init__(self, post_id, post_content):
        self.post_id = post_id                    # 帖子唯一ID
        self.root = CommentNode(post_id, post_content, parent_id=None)  # 帖子本身作为根节点

    def add_comment(self, comment_id, content, parent_id):
        """
        添加评论或回复到树中。
        """
        parent_node = self.find_comment(self.root, parent_id)
        if parent_node:
            new_comment = CommentNode(comment_id, content, parent_id)
            parent_node.add_child(new_comment)
            return new_comment
        else:
            raise ValueError(f"父评论ID {parent_id} 未找到")

    def find_comment(self, node, comment_id):
        """
        递归查找指定ID的评论节点。
        """
        if node.comment_id == comment_id:
            return node
        for child in node.children:
            result = self.find_comment(child, comment_id)
            if result:
                return result
        return None
    
    def set_embeddings(self, embedding_model, batch_size=16):
        """
        为所有节点计算并设置向量表示，利用GPU加速并显示进度条
        
        参数:
            embedding_model: 预训练的嵌入模型
            batch_size: 批处理大小
        """
        
        # 确保模型在GPU上运行(如果可用)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model = embedding_model.to(device)
        
        # 收集所有节点的内容
        all_nodes = []
        all_contents = []
        
        def collect_nodes(node):
            all_nodes.append(node)
            all_contents.append(node.content)
            for child in node.children:
                collect_nodes(child)
        
        collect_nodes(self.root)
        
        # 批量计算嵌入，并使用tqdm显示进度
        all_embeddings = []
        total_batches = (len(all_contents) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(all_contents), batch_size), desc="Calculating embeddings", total=total_batches):
            batch_contents = all_contents[i:i + batch_size]
            
            with torch.no_grad():  # 避免计算梯度
                batch_embeddings = embedding_model.encode(batch_contents)
            
            all_embeddings.extend(batch_embeddings)
        
        # 为每个节点设置嵌入
        for node, embedding in zip(all_nodes, all_embeddings):
            node.set_embedding(embedding)
    
    def prune_by_similarity(self, similarity_threshold=0.5, similarity_func=None):
        """
        基于相似度对一级评论的子树进行剪边，返回森林
        
        参数:
            similarity_threshold: 相似度阈值，低于此值的边将被剪掉
            similarity_func: 自定义相似度计算函数
        
        返回:
            ForestManager对象，包含剪边后的森林
        """
        forest = ForestManager(self)
        forest.similarity_threshold = similarity_threshold
        
        # 遍历所有一级评论（帖子的直接子节点）
        for first_level_comment in self.root.children:
            # 对每个一级评论的子树进行剪边
            pruned_subtree = self._prune_subtree_by_similarity(
                first_level_comment, 
                similarity_threshold, 
                similarity_func,
                forest  # 传递forest对象用于记录剪边
            )
            
            # 将剪边后的主子树添加到森林
            forest.add_subtree(pruned_subtree, self.root.comment_id)
        
        return forest
    
    def _prune_subtree_by_similarity(self, subtree_root, threshold, similarity_func, forest):
        """
        对单个子树进行基于相似度的剪边
        
        参数:
            subtree_root: 子树的根节点
            threshold: 相似度阈值
            similarity_func: 相似度计算函数
            forest: ForestManager对象，用于记录剪边信息
        
        返回:
            剪边后的子树根节点
        """
        # 深度复制子树根节点
        pruned_root = subtree_root.deep_copy()
        
        # 清空子节点，重新构建
        pruned_root.children = []
        
        # 递归处理每个子节点
        for child in subtree_root.children:
            # 计算父子节点之间的相似度
            try:
                similarity = subtree_root.calculate_similarity(child, similarity_func)
                
                if similarity >= threshold:
                    # 相似度足够高，保留边，递归处理子节点
                    pruned_child = self._prune_subtree_by_similarity(
                        child, threshold, similarity_func, forest
                    )
                    pruned_root.add_child(pruned_child)
                else:
                    # 相似度低于阈值，剪掉这条边并记录
                    forest.record_cut_edge(
                        parent_id=subtree_root.comment_id,
                        child_id=child.comment_id,
                        similarity_score=similarity,
                        reason="similarity_below_threshold"
                    )
                    
                    # 将被剪掉的子树作为独立子树添加到森林中
                    orphaned_subtree = self._prune_subtree_by_similarity(
                        child, threshold, similarity_func, forest
                    )
                    forest.add_subtree(orphaned_subtree, subtree_root.comment_id)
            
            except ValueError as e:
                # 如果无法计算相似度（如缺少嵌入），默认保留边但记录原因
                pruned_child = self._prune_subtree_by_similarity(
                    child, threshold, similarity_func, forest
                )
                pruned_root.add_child(pruned_child)
                
                # 记录无法计算相似度的情况
                forest.record_cut_edge(
                    parent_id=subtree_root.comment_id,
                    child_id=child.comment_id,
                    similarity_score=-1,  # 特殊值表示无法计算
                    reason="embedding_missing"
                )
    
        return pruned_root
    
    def analyze_similarity_distribution(self):
        """
        分析树中所有父子节点对的相似度分布
        
        返回:
            相似度分布的统计信息
        """
        similarities = []
        
        def collect_similarities(node):
            for child in node.children:
                try:
                    similarity = node.calculate_similarity(child)
                    similarities.append(similarity)
                except ValueError:
                    pass  # 跳过无法计算相似度的节点对
                collect_similarities(child)
        
        # 只分析一级评论及其子树中的相似度
        for first_level_comment in self.root.children:
            collect_similarities(first_level_comment)
        
        if not similarities:
            return {"error": "无法计算相似度，请确保节点有嵌入表示"}
        
        similarities = np.array(similarities)
        
        return {
            'count': len(similarities),
            'mean': np.mean(similarities),
            'std': np.std(similarities),
            'min': np.min(similarities),
            'max': np.max(similarities),
            'median': np.median(similarities),
            'percentiles': {
                '25': np.percentile(similarities, 25),
                '75': np.percentile(similarities, 75),
                '90': np.percentile(similarities, 90),
                '95': np.percentile(similarities, 95)
            }
        }

class PostStorage:
    """
    管理和存储多个帖子树的数据对象模型。
    """
    def __init__(self):
        self.posts = {}  # {post_id: PostTree} 原始树存储
        self.forests = {}  # {post_id: ForestManager} 森林存储

    def add_post(self, post_id, post_content):
        """
        新增一个帖子。
        """
        if post_id in self.posts:
            raise ValueError(f"帖子ID {post_id} 已存在")
        self.posts[post_id] = PostTree(post_id, post_content)

    def get_post(self, post_id):
        """
        获取指定ID的帖子树。
        """
        return self.posts.get(post_id)

    def add_comment_to_post(self, post_id, comment_id, content, parent_id):
        """
        向指定帖子添加评论或回复。
        """
        post_tree = self.get_post(post_id)
        if not post_tree:
            raise ValueError(f"帖子ID {post_id} 未找到")
        return post_tree.add_comment(comment_id, content, parent_id)
    
    def set_embeddings_for_post(self, post_id, embedding_model, batch_size=16):
        """
        为指定帖子的所有节点设置嵌入表示
        """
        post_tree = self.get_post(post_id)
        if not post_tree:
            raise ValueError(f"帖子ID {post_id} 未找到")
        post_tree.set_embeddings(embedding_model, batch_size)
    
    def set_embeddings_for_all_posts(self, embedding_model, batch_size=16):
        """
        为所有帖子设置嵌入表示
        """
        for post_id in self.posts:
            self.set_embeddings_for_post(post_id, embedding_model, batch_size)
    
    def prune_post_by_similarity(self, post_id, similarity_threshold=0.5, similarity_func=None):
        """
        对指定帖子进行基于相似度的剪边，生成森林
        
        参数:
            post_id: 帖子ID
            similarity_threshold: 相似度阈值
            similarity_func: 自定义相似度函数
        
        返回:
            ForestManager对象
        """
        post_tree = self.get_post(post_id)
        if not post_tree:
            raise ValueError(f"帖子ID {post_id} 未找到")
        
        forest = post_tree.prune_by_similarity(similarity_threshold, similarity_func)
        self.forests[post_id] = forest
        return forest
    
    def get_forest(self, post_id):
        """
        获取指定帖子的森林
        """
        return self.forests.get(post_id)
    
    def get_all_forests(self):
        """
        获取所有森林
        """
        return self.forests.copy()
    
    def analyze_all_similarity_distributions(self):
        """
        分析所有帖子的相似度分布
        
        返回:
            dict: 包含各个帖子分布和总体分布的字典
        """
        all_distributions = {}
        all_similarities = []  # 收集所有帖子的相似度数据
        
        # 分析每个帖子的相似度分布
        for post_id, post_tree in self.posts.items():
            distribution = post_tree.analyze_similarity_distribution()
            all_distributions[post_id] = distribution
            
            # 收集该帖子的相似度数据用于总体统计
            if 'error' not in distribution:
                similarities = []
                
                def collect_similarities(node):
                    for child in node.children:
                        try:
                            similarity = node.calculate_similarity(child)
                            similarities.append(similarity)
                        except ValueError:
                            pass  # 跳过无法计算相似度的节点对
                        collect_similarities(child)
                
                # 只分析一级评论及其子树中的相似度
                for first_level_comment in post_tree.root.children:
                    collect_similarities(first_level_comment)
                
                all_similarities.extend(similarities)
        
        # 计算所有帖子加总的相似度分布
        if all_similarities:
            all_similarities = np.array(all_similarities)
            overall_distribution = {
                'count': len(all_similarities),
                'mean': np.mean(all_similarities),
                'std': np.std(all_similarities),
                'min': np.min(all_similarities),
                'max': np.max(all_similarities),
                'median': np.median(all_similarities),
                'percentiles': {
                    '25': np.percentile(all_similarities, 25),
                    '75': np.percentile(all_similarities, 75),
                    '90': np.percentile(all_similarities, 90),
                    '95': np.percentile(all_similarities, 95)
                }
            }
        else:
            overall_distribution = {"error": "无法计算总体相似度，请确保节点有嵌入表示"}
        
        return {
            'individual_distributions': all_distributions,
            'overall_distribution': overall_distribution,
            'summary': {
                'total_posts': len(self.posts),
                'posts_with_valid_similarities': len([d for d in all_distributions.values() if 'error' not in d]),
                'total_similarity_pairs': len(all_similarities) if len(all_similarities) > 0 else 0
            }
        }

    def print_similarity_distributions_report(self):
        """
        打印所有帖子相似度分布的详细报告
        """
        analysis = self.analyze_all_similarity_distributions()
        
        print("=" * 100)
        print("📊 所有帖子相似度分布分析报告")
        print("=" * 100)
        
        # 总体摘要
        summary = analysis['summary']
        print(f"📋 总体摘要:")
        print(f"   分析帖子数: {summary['total_posts']}")
        print(f"   有效相似度帖子数: {summary['posts_with_valid_similarities']}")
        print(f"   总相似度对数: {summary['total_similarity_pairs']}")
        
        # 总体分布统计
        overall = analysis['overall_distribution']
        if 'error' not in overall:
            print(f"\n🎯 所有帖子加总相似度分布:")
            print(f"   样本数量: {overall['count']}")
            print(f"   平均值: {overall['mean']:.4f}")
            print(f"   标准差: {overall['std']:.4f}")
            print(f"   中位数: {overall['median']:.4f}")
            print(f"   范围: {overall['min']:.4f} - {overall['max']:.4f}")
            print(f"   25th百分位: {overall['percentiles']['25']:.4f}")
            print(f"   75th百分位: {overall['percentiles']['75']:.4f}")
            print(f"   90th百分位: {overall['percentiles']['90']:.4f}")
            print(f"   95th百分位: {overall['percentiles']['95']:.4f}")
        else:
            print(f"\n❌ 总体分布: {overall['error']}")
        
        # 各帖子详细分布（显示前10个）
        individual = analysis['individual_distributions']
        valid_posts = [(post_id, dist) for post_id, dist in individual.items() if 'error' not in dist]
        
        if valid_posts:
            print(f"\n📋 各帖子详细分布 (前10个):")
            print("-" * 100)
            print(f"{'帖子ID':<20} {'样本数':<8} {'平均值':<10} {'标准差':<10} {'中位数':<10} {'范围':<20}")
            print("-" * 100)
            
            for i, (post_id, dist) in enumerate(valid_posts[:10]):
                post_id_short = post_id[:17] + "..." if len(post_id) > 20 else post_id
                range_str = f"{dist['min']:.3f}-{dist['max']:.3f}"
                print(f"{post_id_short:<20} {dist['count']:<8} {dist['mean']:<10.4f} "
                      f"{dist['std']:<10.4f} {dist['median']:<10.4f} {range_str:<20}")
            
            if len(valid_posts) > 10:
                print(f"... 还有 {len(valid_posts) - 10} 个帖子")
        
        # 分布差异分析
        if len(valid_posts) > 1:
            print(f"\n📈 分布差异分析:")
            means = [dist['mean'] for _, dist in valid_posts]
            stds = [dist['std'] for _, dist in valid_posts]
            
            print(f"   各帖子平均值范围: {min(means):.4f} - {max(means):.4f}")
            print(f"   各帖子标准差范围: {min(stds):.4f} - {max(stds):.4f}")
            print(f"   平均值的标准差: {np.std(means):.4f}")
            print(f"   标准差的标准差: {np.std(stds):.4f}")
        
        print("=" * 100)
    
    def prune_all_posts_by_similarity(self, similarity_threshold=0.5, similarity_func=None, show_progress=True):
        """
        对所有帖子进行基于相似度的剪边，生成森林
        
        参数:
            similarity_threshold: 相似度阈值
            similarity_func: 自定义相似度函数
            show_progress: 是否显示进度条
        
        返回:
            dict: {post_id: ForestManager} 所有帖子的森林字典
        """
        if not self.posts:
            print("没有帖子需要剪枝")
            return {}
        
        print(f"\n开始对 {len(self.posts)} 个帖子进行相似度剪枝...")
        print(f"相似度阈值: {similarity_threshold}")
        
        # 使用进度条遍历所有帖子
        post_items = list(self.posts.items())
        if show_progress:
            from tqdm import tqdm
            
            post_items = tqdm(post_items, desc="剪枝帖子", unit="个帖子")
        
        results = {}
        failed_posts = []
        
        for post_id, post_tree in post_items:
            try:
                # 对每个帖子进行剪枝
                forest = post_tree.prune_by_similarity(similarity_threshold, similarity_func)
                self.forests[post_id] = forest
                results[post_id] = forest
                
                if show_progress and not isinstance(post_items, list):
                    # 更新进度条描述
                    stats = forest.get_forest_statistics()
                    post_items.set_postfix({
                        '子树数': stats.get('subtree_count', 0),
                        '剪边数': stats.get('cut_edges_count', 0)
                    })
                    
            except Exception as e:
                failed_posts.append((post_id, str(e)))
                print(f"\n警告: 帖子 {post_id} 剪枝失败: {e}")
        
        # 打印总结
        print(f"\n剪枝完成！")
        print(f"✅ 成功剪枝: {len(results)} 个帖子")
        if failed_posts:
            print(f"❌ 失败: {len(failed_posts)} 个帖子")
            for post_id, error in failed_posts:
                print(f"   - {post_id}: {error}")
        
        return results
    
    def get_all_forest_statistics(self):
        """
        获取所有森林的统计信息
        
        返回:
            dict: 包含总体统计和各个帖子详细统计的字典
        """
        if not self.forests:
            return {"error": "没有森林数据"}
        
        # 收集所有森林的统计信息
        all_stats = {}
        total_subtrees = 0
        total_nodes = 0
        total_cut_edges = 0
        
        for post_id, forest in self.forests.items():
            stats = forest.get_forest_statistics()
            all_stats[post_id] = stats
            
            total_subtrees += stats.get('subtree_count', 0)
            total_nodes += stats.get('total_nodes', 0)
            total_cut_edges += stats.get('cut_edges_count', 0)
        
        # 计算总体统计
        avg_subtrees_per_post = total_subtrees / len(self.forests) if self.forests else 0
        avg_nodes_per_post = total_nodes / len(self.forests) if self.forests else 0
        avg_cut_edges_per_post = total_cut_edges / len(self.forests) if self.forests else 0
        
        return {
            'total_posts': len(self.forests),
            'total_subtrees': total_subtrees,
            'total_nodes': total_nodes,
            'total_cut_edges': total_cut_edges,
            'avg_subtrees_per_post': avg_subtrees_per_post,
            'avg_nodes_per_post': avg_nodes_per_post,
            'avg_cut_edges_per_post': avg_cut_edges_per_post,
            'individual_stats': all_stats
        }
    
    def print_all_forest_statistics(self):
        """
        打印所有森林的统计报告
        """
        stats = self.get_all_forest_statistics()
        
        if 'error' in stats:
            print(stats['error'])
            return
        
        print("=" * 100)
        print("🌳 所有帖子森林统计报告")
        print("=" * 100)
        
        # 总体统计
        print(f"📊 总体统计:")
        print(f"   处理帖子数: {stats['total_posts']}")
        print(f"   总子树数: {stats['total_subtrees']}")
        print(f"   总节点数: {stats['total_nodes']}")
        print(f"   总剪边数: {stats['total_cut_edges']}")
        print(f"   平均每个帖子子树数: {stats['avg_subtrees_per_post']:.2f}")
        print(f"   平均每个帖子节点数: {stats['avg_nodes_per_post']:.2f}")
        print(f"   平均每个帖子剪边数: {stats['avg_cut_edges_per_post']:.2f}")
        
        # 各帖子详细统计（只显示前10个）
        print(f"\n📋 各帖子详细统计 (前10个):")
        print("-" * 100)
        print(f"{'帖子ID':<20} {'子树数':<8} {'节点数':<8} {'剪边数':<8} {'平均子树大小':<12}")
        print("-" * 100)
        
        count = 0
        for post_id, post_stats in stats['individual_stats'].items():
            if count >= 10:
                remaining = len(stats['individual_stats']) - 10
                print(f"... 还有 {remaining} 个帖子")
                break
            
            post_id_short = post_id[:17] + "..." if len(post_id) > 20 else post_id
            print(f"{post_id_short:<20} {post_stats.get('subtree_count', 0):<8} "
                  f"{post_stats.get('total_nodes', 0):<8} {post_stats.get('cut_edges_count', 0):<8} "
                  f"{post_stats.get('avg_subtree_size', 0):<12.2f}")
            count += 1
        
        print("=" * 100)
    
    def analyze_all_forests_distribution(self):
        """
        分析所有森林的节点分布情况
        
        返回:
            综合的分布分析结果
        """
        if not self.forests:
            return {"error": "没有森林数据"}
        
        # 收集所有森林的分布数据
        all_tree_sizes = []
        size_distributions = {'small_trees': 0, 'medium_trees': 0, 'large_trees': 0}
        
        for forest in self.forests.values():
            analysis = forest.analyze_node_distribution()
            
            # 收集所有子树大小
            for subtree in forest.subtrees:
                all_tree_sizes.append(subtree['size'])
            
            # 累计规模分类
            for category, data in analysis['size_distribution'].items():
                size_distributions[category] += data['count']
        
        if not all_tree_sizes:
            return {"error": "没有子树数据"}
        
        # 计算总体统计
        total_subtrees = len(all_tree_sizes)
        total_nodes = sum(all_tree_sizes)
        
        # 计算直方图
        histogram_ranges = [1, 2, 3, 5, 10, 20, 50, 100]
        histogram = {}
        
        for i in range(len(histogram_ranges)):
            if i == 0:
                range_key = f"1"
                count = len([s for s in all_tree_sizes if s == 1])
            elif i == len(histogram_ranges) - 1:
                range_key = f"{histogram_ranges[i]}+"
                count = len([s for s in all_tree_sizes if s >= histogram_ranges[i]])
            else:
                prev_range = histogram_ranges[i-1]
                curr_range = histogram_ranges[i]
                range_key = f"{prev_range+1}-{curr_range}"
                count = len([s for s in all_tree_sizes if prev_range < s <= curr_range])
            
            histogram[range_key] = {
                'count': count,
                'percentage': (count / total_subtrees * 100) if total_subtrees > 0 else 0
            }
        
        # 规模分类百分比
        for category in size_distributions:
            size_distributions[category] = {
                'count': size_distributions[category],
                'percentage': (size_distributions[category] / total_subtrees * 100) if total_subtrees > 0 else 0
            }
        
        return {
            'total_forests': len(self.forests),
            'total_subtrees': total_subtrees,
            'total_nodes': total_nodes,
            'avg_subtree_size': total_nodes / total_subtrees if total_subtrees > 0 else 0,
            'max_subtree_size': max(all_tree_sizes) if all_tree_sizes else 0,
            'min_subtree_size': min(all_tree_sizes) if all_tree_sizes else 0,
            'histogram': histogram,
            'size_distribution': size_distributions
        }
    
    def sample_and_visualize_subtrees_from_distribution(self, samples_per_group=1):
        """
        从不同的直方图分布中随机采样指定数量的子树并可视化
        
        参数:
            samples_per_group: 每个分布区间采样的子树数量，默认为1
        """
        if not self.forests:
            print("❌ 没有森林数据")
            return
        
        # 收集所有子树，按大小分组
        subtree_groups = {
            '1': [],
            '2': [],
            '3': [],
            '4-5': [],
            '6-10': [],
            '11-20': [],
            '21-50': [],
            '51-100': [],
            '100+': []
        }
        
        for post_id, forest in self.forests.items():
            for subtree in forest.subtrees:
                size = subtree['size']
                # 根据大小分组
                if size == 1:
                    subtree_groups['1'].append((post_id, subtree))
                elif size == 2:
                    subtree_groups['2'].append((post_id, subtree))
                elif size == 3:
                    subtree_groups['3'].append((post_id, subtree))
                elif size <= 5:
                    subtree_groups['4-5'].append((post_id, subtree))
                elif size <= 10:
                    subtree_groups['6-10'].append((post_id, subtree))
                elif size <= 20:
                    subtree_groups['11-20'].append((post_id, subtree))
                elif size <= 50:
                    subtree_groups['21-50'].append((post_id, subtree))
                elif size <= 100:
                    subtree_groups['51-100'].append((post_id, subtree))
                else:
                    subtree_groups['100+'].append((post_id, subtree))
        
        print("=" * 120)
        print(f"🎲 从不同分布区间采样子树进行可视化 (每组采样 {samples_per_group} 个)")
        print("=" * 120)
        
        total_sampled = 0
        for group_name, subtrees in subtree_groups.items():
            if not subtrees:
                print(f"\n📊 分布区间 [{group_name}个节点]: 无数据")
                continue
            
            # 确定实际采样数量
            actual_samples = min(samples_per_group, len(subtrees))
            
            print(f"\n{'='*80}")
            print(f"📊 分布区间: [{group_name}个节点] - 总数: {len(subtrees)} 棵，采样: {actual_samples} 棵")
            print(f"{'='*80}")
            
            # 如果采样数量等于总数，则显示所有子树；否则随机采样
            if actual_samples == len(subtrees):
                print(f"💡 该区间子树数量不足 {samples_per_group} 个，显示全部 {len(subtrees)} 棵子树")
                selected_subtrees = subtrees
            else:
                print(f"🎯 从 {len(subtrees)} 棵子树中随机采样 {actual_samples} 棵")
                selected_subtrees = random.sample(subtrees, actual_samples)
            
            # 可视化选中的子树
            for i, (post_id, sampled_subtree) in enumerate(selected_subtrees):
                total_sampled += 1
                
                print(f"\n{'-'*60}")
                print(f"🌳 样本 {i+1}/{actual_samples} (总第 {total_sampled} 个)")
                print(f"{'-'*60}")
                print(f"📍 来源帖子: {post_id}")
                print(f"📊 子树大小: {sampled_subtree['size']} 个节点")
                print(f"📏 子树深度: {sampled_subtree['depth']} 层")
                
                # 可视化这个子树
                print(f"\n🌲 子树结构:")
                self._visualize_subtree_simple(sampled_subtree['root'])
        
        if total_sampled == 0:
            print("❌ 没有找到可采样的子树")
        else:
            print(f"\n✅ 成功采样并可视化了 {total_sampled} 个子树")
            
            # 显示采样统计
            print(f"\n📊 采样统计:")
            for group_name, subtrees in subtree_groups.items():
                if subtrees:
                    actual_samples = min(samples_per_group, len(subtrees))
                    sample_rate = (actual_samples / len(subtrees)) * 100
                    print(f"   {group_name:>8} 节点: {actual_samples}/{len(subtrees)} ({sample_rate:.1f}%)")
        
        print("=" * 120)
        """
        从不同的直方图分布中各随机采样1个子树并可视化
        """
        if not self.forests:
            print("❌ 没有森林数据")
            return
        
        # 收集所有子树，按大小分组
        subtree_groups = {
            '1': [],
            '2': [],
            '3': [],
            '4-5': [],
            '6-10': [],
            '11-20': [],
            '21-50': [],
            '51-100': [],
            '100+': []
        }
        
        for post_id, forest in self.forests.items():
            for subtree in forest.subtrees:
                size = subtree['size']
                # 根据大小分组
                if size == 1:
                    subtree_groups['1'].append((post_id, subtree))
                elif size == 2:
                    subtree_groups['2'].append((post_id, subtree))
                elif size == 3:
                    subtree_groups['3'].append((post_id, subtree))
                elif size <= 5:
                    subtree_groups['4-5'].append((post_id, subtree))
                elif size <= 10:
                    subtree_groups['6-10'].append((post_id, subtree))
                elif size <= 20:
                    subtree_groups['11-20'].append((post_id, subtree))
                elif size <= 50:
                    subtree_groups['21-50'].append((post_id, subtree))
                elif size <= 100:
                    subtree_groups['51-100'].append((post_id, subtree))
                else:
                    subtree_groups['100+'].append((post_id, subtree))
        
        print("=" * 120)
        print("🎲 从不同分布区间随机采样子树进行可视化")
        print("=" * 120)
        
        
        sampled_count = 0
        for group_name, subtrees in subtree_groups.items():
            if not subtrees:
                print(f"\n📊 分布区间 [{group_name}个节点]: 无数据")
                continue
            
            # 随机选择一个子树
            post_id, sampled_subtree = random.choice(subtrees)
            sampled_count += 1
            
            print(f"\n{'='*80}")
            print(f"🌳 样本 {sampled_count} - 分布区间: [{group_name}个节点]")
            print(f"{'='*80}")
            print(f"📍 来源帖子: {post_id}")
            print(f"📊 子树大小: {sampled_subtree['size']} 个节点")
            print(f"📏 子树深度: {sampled_subtree['depth']} 层")
            print(f"🔢 该区间总数: {len(subtrees)} 棵子树")
            
            # 可视化这个子树
            print(f"\n🌲 子树结构:")
            self._visualize_subtree_simple(sampled_subtree['root'])
        
        if sampled_count == 0:
            print("❌ 没有找到可采样的子树")
        else:
            print(f"\n✅ 成功采样并可视化了 {sampled_count} 个不同分布区间的子树")
        print("=" * 120)
    
    def _visualize_subtree_simple(self, node, depth=0, prefix=""):
        """
        简化的子树可视化方法
        """
        # 显示当前节点
        content_preview = node.content[:60] + '...' if len(node.content) > 60 else node.content
        node_info = f"[L{depth}] {content_preview}"
        print(f"{prefix}{node_info}")
        print(f"{prefix}     ID: {node.comment_id}")
        
        # 显示子节点
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            
            if is_last_child:
                branch = "└─── "
                child_prefix = prefix + "     "
            else:
                branch = "├─── "
                child_prefix = prefix + "│    "
            
            print(f"{prefix}{branch}")
            self._visualize_subtree_simple(child, depth + 1, child_prefix)
    
    def compare_forests_by_threshold(self, thresholds=[0.3, 0.5, 0.7, 0.9], post_id=None):
        """
        比较不同阈值下的剪枝效果
        
        参数:
            thresholds: 要比较的相似度阈值列表
            post_id: 指定帖子ID，如果为None则使用第一个帖子
        
        返回:
            不同阈值下的比较结果
        """
        if not self.posts:
            return {"error": "没有帖子数据"}
        
        # 选择要分析的帖子
        if post_id is None:
            post_id = list(self.posts.keys())[0]
        
        if post_id not in self.posts:
            return {"error": f"帖子 {post_id} 不存在"}
        
        post_tree = self.posts[post_id]
        comparison_results = {}
        
        print(f"\n比较帖子 {post_id} 在不同阈值下的剪枝效果:")
        print("-" * 80)
        print(f"{'阈值':<8} {'子树数':<8} {'总节点数':<10} {'剪边数':<8} {'平均子树大小':<12}")
        print("-" * 80)
        
        for threshold in thresholds:
            try:
                forest = post_tree.prune_by_similarity(threshold)
                stats = forest.get_forest_statistics()
                
                comparison_results[threshold] = stats
                
                print(f"{threshold:<8} {stats.get('subtree_count', 0):<8} "
                      f"{stats.get('total_nodes', 0):<10} {stats.get('cut_edges_count', 0):<8} "
                      f"{stats.get('avg_subtree_size', 0):<12.2f}")
                      
            except Exception as e:
                print(f"{threshold:<8} 错误: {e}")
        
        print("-" * 80)
        return comparison_results

    def visualize_forest_with_cut_edges(self, post_id=None, max_depth=None, max_subtrees=None, show_cut_edges=True, min_subtree_size=2):
        """
        可视化森林结构，显示子树和被剪掉的边信息
        
        参数:
            post_id: 要可视化的帖子ID，如果为None则使用第一个森林
            max_depth: 显示的最大深度
            max_subtrees: 显示的最大子树数量
            show_cut_edges: 是否显示被剪掉的边信息
            min_subtree_size: 显示子树的最小节点数量，默认为1
        """
        if not self.forests:
            print("❌ 没有森林数据可供可视化")
            return
        
        # 选择要可视化的森林
        if post_id is None:
            post_id = list(self.forests.keys())[0]
        
        if post_id not in self.forests:
            print(f"❌ 帖子 {post_id} 的森林不存在")
            return
        
        forest = self.forests[post_id]
        
        print("=" * 120)
        print(f"🌲 帖子 {post_id} 剪枝后森林结构可视化")
        print("=" * 120)
        
        # 显示基本统计信息
        stats = forest.get_forest_statistics()
        print(f"📊 森林统计: {stats.get('subtree_count', 0)} 棵子树, "
              f"{stats.get('total_nodes', 0)} 个节点, "
              f"{stats.get('cut_edges_count', 0)} 条被剪边")
        print(f"🎯 相似度阈值: {forest.similarity_threshold}")
        
        # 过滤满足最小大小要求的子树
        filtered_subtrees = [subtree for subtree in forest.subtrees if subtree['size'] >= min_subtree_size]
        
        if not filtered_subtrees:
            print(f"❌ 没有找到节点数 >= {min_subtree_size} 的子树")
            return
        
        print(f"🔍 筛选条件: 显示节点数 >= {min_subtree_size} 的子树")
        print(f"📊 筛选结果: {len(filtered_subtrees)}/{len(forest.subtrees)} 棵子树符合条件")
        
        # 构建被剪边的映射关系
        cut_edge_map = {}
        for edge in forest.cut_edges:
            parent_id = edge['parent_id']
            if parent_id not in cut_edge_map:
                cut_edge_map[parent_id] = []
            cut_edge_map[parent_id].append(edge)
        
        # 显示子树
        subtrees_to_show = filtered_subtrees[:max_subtrees] if max_subtrees else filtered_subtrees
        
        for i, subtree_info in enumerate(subtrees_to_show):
            print(f"\n{'='*60}")
            print(f"🌳 子树 {i+1}/{len(filtered_subtrees)}")
            print(f"{'='*60}")
            print(f"📍 根节点ID: {subtree_info['root'].comment_id}")
            print(f"📊 大小: {subtree_info['size']} 个节点")
            print(f"📏 深度: {subtree_info['depth']} 层")
            print(f"🔗 原始父节点: {subtree_info['original_parent_id']}")
            
            # 显示从这个节点被剪掉的边
            root_id = subtree_info['root'].comment_id
            if show_cut_edges and root_id in cut_edge_map:
                print(f"\n✂️  从此节点剪掉的边:")
                for edge in cut_edge_map[root_id]:
                    similarity = edge['similarity_score']
                    child_id = edge['child_id']
                    reason = edge['reason']
                    print(f"   ┈┈┈ {root_id} ┈┈┈▷ {child_id} (相似度: {similarity:.4f}, 原因: {reason})")
            
            print(f"\n🌲 子树结构:")
            self._visualize_subtree(subtree_info['root'], max_depth, cut_edge_map, show_cut_edges)
        
        # 如果有更多子树未显示
        if max_subtrees and len(filtered_subtrees) > max_subtrees:
            remaining = len(filtered_subtrees) - max_subtrees
            print(f"\n... 还有 {remaining} 棵符合条件的子树未显示")
        
        # 显示被剪边的总体统计
        if show_cut_edges and forest.cut_edges:
            self._show_cut_edges_summary(forest)
        
        print("=" * 120)
    
    def _visualize_subtree(self, node, max_depth=None, cut_edge_map=None, show_cut_edges=True, depth=0, prefix=""):
        """
        递归可视化单个子树
        """
        if max_depth is not None and depth > max_depth:
            return
        
        # 显示当前节点
        content_preview = node.content[:50] + '...' if len(node.content) > 50 else node.content
        node_type = f"L{depth}" if depth > 0 else "ROOT"
        print(f"{prefix}[{node_type}] {content_preview}")
        print(f"{prefix}     ID: {node.comment_id}")
        
        # 显示子节点和连接关系
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            
            # 连接线样式
            if is_last_child:
                branch = "└─── "
                child_prefix = prefix + "     "
            else:
                branch = "├─── "
                child_prefix = prefix + "│    "
            
            # 显示连接关系
            print(f"{prefix}{branch}[保留连接] ▶")
            
            # 递归显示子节点
            self._visualize_subtree(child, max_depth, cut_edge_map, show_cut_edges, 
                                  depth + 1, child_prefix)
        
        # 显示从当前节点被剪掉的边
        if show_cut_edges and cut_edge_map and node.comment_id in cut_edge_map:
            cut_edges = cut_edge_map[node.comment_id]
            for j, edge in enumerate(cut_edges):
                is_last_cut = (j == len(cut_edges) - 1) and (len(node.children) == 0)
                
                if len(node.children) == 0 and is_last_cut:
                    cut_branch = "└┈┈┈ "
                else:
                    cut_branch = "├┈┈┈ "
                
                similarity = edge['similarity_score']
                child_id = edge['child_id']
                print(f"{prefix}{cut_branch}[已剪断] ▷ 节点 {child_id} (相似度: {similarity:.4f})")
    
    def _show_cut_edges_summary(self, forest):
        """
        显示被剪边的汇总信息
        """
        print(f"\n{'='*60}")
        print("✂️  被剪边汇总信息")
        print(f"{'='*60}")
        
        # 按相似度分组统计
        similarity_ranges = {
            '0.0-0.1': 0, '0.1-0.2': 0, '0.2-0.3': 0, '0.3-0.4': 0, '0.4-0.5': 0,
            '0.5-0.6': 0, '0.6-0.7': 0, '0.7-0.8': 0, '0.8-0.9': 0, '0.9-1.0': 0
        }
        
        valid_similarities = []
        for edge in forest.cut_edges:
            sim = edge['similarity_score']
            if sim >= 0:  # 排除无法计算相似度的边
                valid_similarities.append(sim)
                # 分组统计
                if sim < 0.1:
                    similarity_ranges['0.0-0.1'] += 1
                elif sim < 0.2:
                    similarity_ranges['0.1-0.2'] += 1
                elif sim < 0.3:
                    similarity_ranges['0.2-0.3'] += 1
                elif sim < 0.4:
                    similarity_ranges['0.3-0.4'] += 1
                elif sim < 0.5:
                    similarity_ranges['0.4-0.5'] += 1
                elif sim < 0.6:
                    similarity_ranges['0.5-0.6'] += 1
                elif sim < 0.7:
                    similarity_ranges['0.6-0.7'] += 1
                elif sim < 0.8:
                    similarity_ranges['0.7-0.8'] += 1
                elif sim < 0.9:
                    similarity_ranges['0.8-0.9'] += 1
                else:
                    similarity_ranges['0.9-1.0'] += 1
        
        print(f"📊 总被剪边数: {len(forest.cut_edges)}")
        if valid_similarities:
            print(f"📈 相似度统计:")
            print(f"   平均值: {np.mean(valid_similarities):.4f}")
            print(f"   中位数: {np.median(valid_similarities):.4f}")
            print(f"   范围: {min(valid_similarities):.4f} - {max(valid_similarities):.4f}")
            
            print(f"\n📊 相似度分布:")
            for range_name, count in similarity_ranges.items():
                if count > 0:
                    percentage = count / len(valid_similarities) * 100
                    bar = "█" * min(20, count)
                    print(f"   {range_name}: {bar} {count} ({percentage:.1f}%)")
        
        # 显示剪边原因统计
        reason_counts = {}
        for edge in forest.cut_edges:
            reason = edge['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        print(f"\n🔍 剪边原因统计:")
        for reason, count in reason_counts.items():
            percentage = count / len(forest.cut_edges) * 100
            print(f"   {reason}: {count} ({percentage:.1f}%)")
    
    def visualize_original_vs_pruned(self, original_tree, max_depth=None):
        """
        对比显示原始树和剪枝后的森林
        
        参数:
            original_tree: 原始PostTree对象
            max_depth: 显示的最大深度
        """
        print("=" * 140)
        print("🔄 原始树 vs 剪枝后森林对比")
        print("=" * 140)
        
        # 构建被剪边的集合
        cut_edges = set()
        for edge in self.cut_edges:
            cut_edges.add((edge['parent_id'], edge['child_id']))
        
        print("📊 对比统计:")
        original_stats = self._count_original_tree_stats(original_tree.root)
        forest_stats = self.get_forest_statistics()
        
        print(f"   原始树: {original_stats['total_nodes']} 个节点, {original_stats['total_edges']} 条边")
        print(f"   剪枝后: {forest_stats.get('total_nodes', 0)} 个节点, "
              f"{original_stats['total_edges'] - len(self.cut_edges)} 条边, "
              f"{forest_stats.get('subtree_count', 0)} 棵子树")
        print(f"   被剪掉: {len(self.cut_edges)} 条边 "
              f"({len(self.cut_edges)/original_stats['total_edges']*100:.1f}%)")
        
        print(f"\n🌳 原始树结构 (标记被剪边):")
        print("-" * 70)
        self._visualize_original_with_cuts(original_tree.root, cut_edges, max_depth)
