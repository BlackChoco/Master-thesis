import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
import jieba
import json
import os
from modelscope import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig


def load_model_from_modelscope(model_name_or_path: str, trust_remote_code: bool = True, **kwargs):
    """
    优先从ModelScope加载模型，失败时提示用户
    """
    try:
        print(f"  正在从ModelScope加载模型: {model_name_or_path}")
        model = AutoModel.from_pretrained(
            model_name_or_path, 
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        print(f"  成功从ModelScope加载模型: {model_name_or_path}")
        return model
    except Exception as e:
        print(f"  从ModelScope加载模型失败: {model_name_or_path}")
        print(f"错误信息: {e}")
        print("请检查模型名称是否正确，或网络连接是否正常。")
        raise e


def load_tokenizer_from_modelscope(model_name_or_path: str, trust_remote_code: bool = True, **kwargs):
    """
    优先从ModelScope加载分词器，失败时提示用户
    """
    try:
        print(f"  正在从ModelScope加载分词器: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        print(f"  成功从ModelScope加载分词器: {model_name_or_path}")
        return tokenizer
    except Exception as e:
        print(f"  从ModelScope加载分词器失败: {model_name_or_path}")
        print(f"错误信息: {e}")
        print("请检查模型名称是否正确，或网络连接是否正常。")
        raise e


class TextCNNTokenizer:
    """
    使用预构建词汇表和jieba的TextCNN模型分词器。
    """
    def __init__(self, word_to_idx: Dict[str, int], max_length: int = 128):
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_token_id = word_to_idx.get(self.pad_token, 0)
        self.unk_token_id = word_to_idx.get(self.unk_token, 1)

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            text = str(text)
        return jieba.lcut(text.strip())

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.word_to_idx.get(token, self.unk_token_id) for token in tokens]

    def _pad_truncate_single(self, token_ids: List[int], max_len: int) -> tuple[List[int], List[int]]:
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
                current_len = len(token_ids)
                attention_mask = [1] * current_len
                if current_len < effective_max_length:
                    pad_len = effective_max_length - current_len
                    token_ids.extend([self.pad_token_id] * pad_len)
                    attention_mask.extend([0] * pad_len)
                elif current_len > effective_max_length:
                    token_ids = token_ids[:effective_max_length]
                    attention_mask = [1] * effective_max_length

                padded_ids = token_ids
            else:
                padded_ids = token_ids 
                attention_mask = [1] * len(token_ids)

            batch_input_ids.append(padded_ids)
            batch_attention_masks.append(attention_mask)

        if return_tensors == 'pt':
            if not padding:
                max_len_in_batch = 0
                if batch_input_ids:
                    max_len_in_batch = max(len(ids) for ids in batch_input_ids) if batch_input_ids else 0
                
                if max_len_in_batch > 0:
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
                raise e

            return {
                'input_ids': batch_input_ids_tensor,
                'attention_mask': batch_attention_masks_tensor
            }
        else:
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
        self.fc = torch.nn.Linear(len(filter_sizes) * num_filters, output_dim_for_encoder)
        self.base_dim = output_dim_for_encoder

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


class ContrastiveEncoder(torch.nn.Module):
    """
    修改后的对比编码器：支持ModelScope AutoModel和自定义TextCNN。
    """
    def __init__(self, model_type: str,
                 model_name_or_path: Optional[str] = None,
                 vocab: Optional[Dict[str, int]] = None,
                 textcnn_config: Optional[Dict] = None,
                 projection_hidden_dim: int = 512, 
                 projection_output_dim: int = 256, 
                 projection_dropout_rate: float = 0.1):
        super().__init__()
        self.model_type = model_type.lower()
        self.tokenizer = None
        self.base_model = None
        self.base_dim = 0

        if self.model_type == 'modelscope':
            if model_name_or_path is None:
                raise ValueError("对于 'modelscope' 模型类型，model_name_or_path 是必需的")
            
            # 使用ModelScope加载模型和分词器
            self.tokenizer = load_tokenizer_from_modelscope(model_name_or_path)
            self.base_model = load_model_from_modelscope(
                model_name_or_path,
                torch_dtype=torch.float32
            )
            
            if hasattr(self.base_model.config, 'hidden_size'):
                 self.base_dim = self.base_model.config.hidden_size
            elif hasattr(self.base_model.config, 'd_model'):
                 self.base_dim = self.base_model.config.d_model
            else:
                try:
                    dummy_input = self.tokenizer("test", return_tensors="pt")
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

            print(f"  ModelScope ContrastiveEncoder 初始化完成:")
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
            print(f"  TextCNN ContrastiveEncoder 初始化完成:")
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
        ).float()
        
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

    def _ensure_list_of_strings(self, texts: Union[str, List]) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = []
        for text_input in texts:
            if not isinstance(text_input, str):
                processed_texts.append(str(text_input) if text_input is not None else "")
            else:
                processed_texts.append(text_input if text_input else "")
        return processed_texts

    def get_base_embeddings(self, texts: Union[str, List]):
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

    def forward(self, texts: Union[str, List]):
        """
        定义模型的完整前向传播：基础模型 -> 投影头。
        """
        base_embeddings = self.get_base_embeddings(texts)
        
        if base_embeddings.nelement() == 0:
            proj_output_dim = self.projection_head[-2].out_features
            return torch.empty(0, proj_output_dim, device=base_embeddings.device, dtype=torch.float)
            
        projected_embeddings = self.projection_head(base_embeddings)
        return projected_embeddings

    def save_base_model(self, path: str):
        """保存基础模型 (ModelScope 或 TextCNN) 及其分词器/词汇表。"""
        os.makedirs(path, exist_ok=True)
        if self.model_type == 'modelscope':
            self.base_model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"  ModelScope 基础模型和分词器已保存到: {path}")
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

            print(f"  TextCNN 基础模型 state_dict 已保存到: {model_save_path}")
            print(f"  TextCNN 词汇表已保存到: {vocab_save_path}")
            print(f"  TextCNN 分词器配置已保存到: {tokenizer_config_path}")
        else:
            print(f"  未知模型类型 '{self.model_type}'，无法保存基础模型。")