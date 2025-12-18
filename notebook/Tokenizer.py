from typing import Iterable, Iterator
import json
from collections import Counter
import regex as re

def flatten_to_bytes(items:Iterable):
    res=[]
    for item in items:
        if isinstance(item,Iterable):
            res.extend(item)
        else:
            res.append(item)
    return bytes(res)

class Tokenizer():
    
    PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def __init__(self, vocab:dict[int,bytes]={}, merges:list[tuple[bytes,bytes]]={}, special_tokens:list[str]=None):
        self.vocab=vocab
        self.merges=merges
        if special_tokens:
            self.special_tokens=special_tokens
        else:
            self.special_tokens=[]
        self.vocab=self.__set_special_token_vocab__()
        self.merges_rank_dict=self.__set_merges_rank__()
        self.reverse_vocab=self.__set_reverse_vocab__()
    
    def __set_special_token_vocab__(self):
        if self.special_tokens:
            special_token_dict={}
            raw_vocab_size=len(self.vocab)
            for st in self.special_tokens:
                    st_bytes=st.encode('utf-8')
                    if st_bytes not in self.vocab.values():
                        special_token_dict[len(special_token_dict)+raw_vocab_size]=st_bytes
            for (vocab_id,vocab_bytes) in special_token_dict:
                self.vocab[vocab_id]=vocab_bytes
        return self.vocab
    
    def __set_merges_rank__(self):
        merges_rank_dict={}
        merges_len=len(self.merges)
        for i in range(merges_len):
            merges_rank_dict[self.merges[i]]=i
        return merges_rank_dict
    
    def __set_reverse_vocab__(self):
        return dict(zip(self.vocab.values(),self.vocab.keys()))
    
    def flatten_to_bytes(self,items:Iterable):
        res=[]
        for item in items:
            if isinstance(item,Iterable):
                res.extend(item)
            else:
                res.append(item)
        return bytes(res)
    
    @classmethod
    def get_bytes_map(cls)->(dict[int,str],dict[str,int]):
        bs=list(range(ord('!'),ord('~')+1))+list(range(ord('¡'),ord('¬')+1))+list(range(ord("®"), ord("ÿ")+1))
        cs=bs[:]
        diff=0
        for idx in range(2**8):
            if idx not in bs:
                bs.append(idx)
                cs.append(diff+2**8)
                diff+=1
        cs=[chr(c) for c in cs]
        byte_2_chr=dict(zip(bs,cs))
        chr_2_byte=dict(zip(cs,bs))
        return byte_2_chr,chr_2_byte          
    
    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens:list[str]=None):
        new_vocab={}
        new_merges=[]
        byte_2_chr,chr_2_byte=cls.get_bytes_map()
        with open(vocab_filepath,"r",encoding='utf-8') as vocab_file:
            vocab_str=vocab_file.read()
            vocab_load=json.loads(vocab_str)
            new_vocab={}
            for item in vocab_load.items():
                token_byte_chrs=item[0]
                token_id=item[1]
                token_bytes=bytes([chr_2_byte[byte_chr] for byte_chr in token_byte_chrs])
                new_vocab[token_id]=token_bytes
            for st in special_tokens:
                st_bytes=st.encode('utf-8')
                if st_bytes not in vocab_load.keys():
                    new_vocab[len(new_vocab)]=st_bytes
        with open(merges_filepath,"r",encoding='utf-8') as merges_file:
            new_merges=[
                tuple([
                    bytes([chr_2_byte[byte_chr] for byte_chr in line_str.strip().split(' ')[0]]),
                   bytes([chr_2_byte[byte_chr] for byte_chr in line_str.strip().split(' ')[1]])
            ]) for line_str in merges_file.readlines()]
        return Tokenizer(new_vocab,new_merges,special_tokens)
    
    @classmethod
    def pre_tokenizer(cls, raw_str:str, special_tokens:list[str]):
        """将一段完整文本拆分为 (特殊token切分)+(常见拆词方法)的两级嵌套list结构:[[str]]
        """
        parts=[]
        if special_tokens:        
            special_tokens_sorted=sorted(set(special_tokens),key=len,reverse=True)
            split_pat=r"("+"|".join(map(re.escape,special_tokens_sorted))+r")"
            parts = re.split(split_pat,raw_str)
        else:
            parts = [raw_str]
        result=[]
        for part in parts:
            if part!='' and part not in special_tokens:
                result.append(re.findall(cls.PAT,part))
            elif part!='' and part in special_tokens:
                result.append([part])
        return result
    
    @classmethod
    def pre_tokenizer_for_train(cls, raw_str:str, special_tokens:list[str]):
        """将一段完整文本拆分为 (特殊token切分)+(常见拆词方法)的两级嵌套list结构:[[str]]
        """
        parts=[]
        if special_tokens:        
            special_tokens_sorted=sorted(set(special_tokens),key=len,reverse=True)
            split_pat=r"("+"|".join(map(re.escape,special_tokens_sorted))+r")"
            parts = re.split(split_pat,raw_str)
        else:
            parts = [raw_str]
        result=[]
        for part in parts:
            if part!='' and part not in special_tokens:
                result.append(re.findall(cls.PAT,part))
        return result
    
    @classmethod
    def __init_train_vocab__(cls,special_tokens:list[str]):
        # # 安全的byte取值范围
        # safe_bytes=list(range(ord('!'),ord('~')+1))+list(range(ord('¡'),ord('¬')+1))+list(range(ord("®"), ord("ÿ")+1))
        # base_bytes=safe_bytes[:]
        # # 补充上非安全byte的偏移量
        # diff=0
        # for idx in range(2**8):
        #     if idx not in safe_bytes:
        #         ## CS336不支持test不支持这样
        #         base_bytes.append(2**8+diff)
        #         diff+=1
        # # 构造各位置上映射的unicode编码
        # vocab={}
        # for i in range(len(base_bytes)):
        #     vocab[i]=bytes([base_bytes[i]])
        vocab={i:bytes([i]) for i in range(2**8)}
        # 补充上special_tokens
        for st in special_tokens:
            vocab[len(vocab)]=st.encode('utf-8')
        return vocab
    
    @classmethod
    def __init_word_and_pair_freq__(cls,chunk_word_arr:list[list[str]]):
        word_arr:list[str]=[]
        for words in chunk_word_arr:
            word_arr.extend(words)
        word_bytes_freq = Counter([
            tuple(bytes([b]) for b in x.encode('utf-8'))
            for x in word_arr
        ])
        pair_freq={}
        for word_bytes in word_bytes_freq.keys():
            for pair_byte in zip(word_bytes,word_bytes[1:]):
                pair_freq[pair_byte]=pair_freq.get(pair_byte,0)+word_bytes_freq[word_bytes]
        return word_bytes_freq,pair_freq
    
    @classmethod
    def __init_train_variables__(cls,raw_str:str,special_tokens:list[str]):
        # 构建vocab
        vocab=cls.__init_train_vocab__(special_tokens)
        # 预处理训练数据
        chunk_word_arr=cls.pre_tokenizer_for_train(raw_str,special_tokens)
        # 统计分词后词频
        word_bytes_freq,pair_freq=cls.__init_word_and_pair_freq__(chunk_word_arr)
        return vocab,word_bytes_freq,pair_freq
    
    @classmethod
    def update_pair_freq(cls,word_bytes_freq:Counter,pair_freq:Counter,max_freq_pair:tuple):
        """
        作用：上一轮merge操作造成了对词频的影响，需要更新pair_freq
        时机：max_freq_pair对于word_bytes_freq的合并已经在更新pair_freq之前
        Args:
            max_freq_pair: (b'o',b'w')
            word_bytes_freq (Counter): [(b'l', b'ow'), (b' ', b'l', b'ow'),... ]
            pair_freq (Counter): [[(b' ', b'l'),(b'l', b'o'),(b'o', b'w')],....]--->[[(b' ', b'l'),(b'l', b'ow')],....]
        """
        max_freq_pair_bytes=flatten_to_bytes(max_freq_pair)
        # 首先保证当前word包含了max_freq_pair，这样才有可能需要做的可行性
        for word_bytes in word_bytes_freq.keys():
            if max_freq_pair_bytes not in word_bytes:
                continue
            ## 检查前后序列是否
            for i in range(len(word_bytes)):
                if (i==(len(word_bytes)-1)) or (word_bytes[i]!=max_freq_pair_bytes and word_bytes[i+1]!=max_freq_pair_bytes):
                    continue
                elif (word_bytes[i]!=max_freq_pair_bytes and word_bytes[i+1]==max_freq_pair_bytes):
                    old_pair=(word_bytes[i],max_freq_pair[0],)
                    new_pair=(word_bytes[i],max_freq_pair_bytes,)
                elif (word_bytes[i]==max_freq_pair_bytes and word_bytes[i+1]!=max_freq_pair_bytes):
                    old_pair=(max_freq_pair[1],word_bytes[i+1],)
                    new_pair=(max_freq_pair_bytes,word_bytes[i+1],)
                else:
                    old_pair=(max_freq_pair[1],max_freq_pair[0],)
                    new_pair=(max_freq_pair_bytes,max_freq_pair_bytes,)
                ## 更新旧的old_pair频次
                pair_freq[old_pair]=pair_freq[old_pair]-word_bytes_freq[word_bytes]
                ## 更新新的new_pair频次
                pair_freq[new_pair]=pair_freq.get(new_pair,0)+word_bytes_freq[word_bytes]
        return pair_freq
    
    @classmethod
    def merge_max_pair(cls,word_bytes_freq:Counter,max_freq_pair:tuple,pair_freq:Counter):
        """根据max_freq_pair，更新原有的word_bytes分割、合并方式
        Args:
            max_freq_pair (tuple): (b'o',b'w')
            word_bytes_freq (Counter): [(b'l', b'o', b'w'), (b' ', b'l', b'o', b'w'),... ]-> [(b'l', b'ow'), (b' ', b'l', b'ow'),... ]
        """
        # new_word_bytes_freq=word_bytes_freq.copy()
        word_bytes_list=list(word_bytes_freq.keys())
        max_freq_pair_bytes=flatten_to_bytes(max_freq_pair)
        for word_bytes in word_bytes_list:
            new_word_bytes=[]
            i=0
            bytes_len=len(word_bytes)
            while i < bytes_len:
                if i == len(word_bytes)-1:
                    # 只剩余1位的情况处理
                    new_word_bytes.append(word_bytes[i])
                    i+=1
                    continue
                if word_bytes[i]==max_freq_pair[0] and word_bytes[i+1]==max_freq_pair[1]:
                    new_word_bytes.append(max_freq_pair_bytes)
                    pair_freq[max_freq_pair]=pair_freq[max_freq_pair]-word_bytes_freq[word_bytes]
                    i+=1
                else:
                    new_word_bytes.append(word_bytes[i])
                i+=1
            temp=word_bytes_freq[word_bytes]
            del word_bytes_freq[word_bytes]
            word_bytes_freq[tuple(new_word_bytes)]=temp
        return word_bytes_freq

    @classmethod
    def train_bpe(cls,vocab:dict,word_bytes_freq:Counter,pair_freq:Counter,vocab_max_limit:int):
        merge_list=[]
        max_freq_pair=None
        while(len(vocab)<vocab_max_limit):
            if max_freq_pair:
                ## 由max_freq_pair的新合并现状，更新pair_freq
                pair_freq=cls.update_pair_freq(word_bytes_freq,pair_freq,max_freq_pair)
            ## 找到pair_freq更新后[新的max_freq_pair]
            max_freq_pair=max(pair_freq,key=lambda x:(pair_freq[x],x))
            if pair_freq[max_freq_pair]==0:
                return vocab,merge_list
            max_freq_pair_bytes=flatten_to_bytes(max_freq_pair)
            vocab[len(vocab)]=max_freq_pair_bytes
            # merge_pair_str=(
            #     ''.join([byte_2_chr[b] for b in max_freq_pair[0]]),
            #     ''.join([byte_2_chr[b] for b in max_freq_pair[1]])
            # )
            merge_list.append(max_freq_pair)
            ## 对word_bytes_freq中涉及[新的max_freq_pair]做合并操作
            word_bytes_freq=cls.merge_max_pair(word_bytes_freq,max_freq_pair,pair_freq)
        return vocab,merge_list

    def pre_token_encode(self, text:str)->list[int]:
        # print("DEBUG-pre_token_encode-text",text)
        ## init
        word_bytes=tuple([bytes([x]) for x in text.encode('utf-8')])
        ## 循环，找到最大的
        while True:
            # 统计当前pair对中，优先级最高的
            # <word中byte的位置，匹配到的merge规则序号>
            merge_idx_rank={}
            curr_word_bytes_len=len(word_bytes)
            for i in range(curr_word_bytes_len-1):
                bytes_pair=(word_bytes[i],word_bytes[i+1])
                if bytes_pair in self.merges_rank_dict.keys():
                    merge_idx_rank[i]=self.merges_rank_dict[bytes_pair]
            if not merge_idx_rank:
                break
            min_item=min(merge_idx_rank.items(),key=lambda x:x[1])
            min_idx=min_item[0]
            new_word_bytes=tuple()
            i=0
            while i < curr_word_bytes_len:
                if i == min_idx:
                    new_word_bytes+=(word_bytes[i]+word_bytes[i+1],)
                    i+=1
                else:
                    new_word_bytes+=(word_bytes[i],)
                i+=1
            word_bytes=new_word_bytes
        ## 合并结束后，逐token翻译
        result=[self.reverse_vocab.get(item) for item in word_bytes]
        return result
    
    def encode(self, text: str) -> list[int]:
        result=[]
        if not text:
            return result
        parts=self.pre_tokenizer(text,self.special_tokens)
        # print("DEBUG-encode-parts",parts)
        for part in parts:
            for chunk in part:
                # print("DEBUG-encode-chunk",chunk)
                chunk_bytes=chunk.encode('utf-8')
                # print("DEBUG-encode-chunk_bytes",chunk_bytes)
                # print("DEBUG-encode-self.vocab.keys()",[key for key in self.vocab.keys()][:200])
                if chunk_bytes in self.reverse_vocab.keys():
                    result.append(self.reverse_vocab.get(chunk_bytes))
                else:
                    vocab_ids=self.pre_token_encode(chunk)
                    result.extend(vocab_ids)
        # print("DEBUG-encode-result",result)
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids=self.encode(text)
            for token_id in token_ids:
                yield token_id
        return
                
    def decode(self, ids: list[int]) -> str:
        ## encode: text(拆解)->byte->map_chr(聚合)->vocab_id
        ## decode: vocab_id->map_chr->byte(聚合)->text
        vocab_bytes_list=[self.vocab.get(id) for id in ids]
        bytes_buffer=b''
        for vocab_bytes in vocab_bytes_list:
            bytes_buffer+=vocab_bytes
        return bytes_buffer.decode('utf-8',errors="replace")

# reference_vocab_path = "/root/CS336/assignment1-basics/tests/fixtures/train-bpe-reference-vocab.json"
# byte_2_chr,gpt2_byte_decoder=Tokenizer.get_bytes_map()
# gpt2_reference_vocab={}
# reference_vocab={}
# with open(reference_vocab_path, encoding="utf-8") as f:
#             gpt2_reference_vocab = json.load(f)
#             reference_vocab = {
#                 gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
#                 for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
#             }
# pass

# vocab_filepath="/root/CS336/assignment1-basics/tests/fixtures/gpt2_vocab.json"
# merges_filepath="/root/CS336/assignment1-basics/tests/fixtures/gpt2_merges.txt"
# address_filepath="/root/CS336/assignment1-basics/tests/fixtures/address.txt"
# address_raw_str=""
# with open(address_filepath) as f:
#     address_raw_str=f.read()

# special_tokens=["<|endoftext|>"]
# special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]
# special_tokens=[]
# temp=Tokenizer.__init_train_vocab__(special_tokens)


# t=Tokenizer.from_files(vocab_filepath,merges_filepath,special_tokens)

# with open("/root/CS336/assignment1-basics/tests/fixtures/tinystories_sample.txt") as f:
#     all_ids=[]
#     for _id in t.encode_iterable(f):
#         all_ids.append(_id)

# origin_text="I come from china，你呢"
# origin_text="Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
# origin_text="Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
# origin_text=address_raw_str
# encode_ids=t.encode(origin_text)
# decode_str=t.decode(encode_ids)

# print(origin_text)
# print(encode_ids)
# print(decode_str)


# print(t.split_by_special_tokens("ab cPAT1 defPAT2PAT3xyzPAT1PAT1 end    ",["PAT1", "PAT2", "PAT3"]))

# with open('/root/CS336/assignment1-basics/tests/fixtures/corpus.en') as f:
#     raw_str=f.read()
#     special_tokens=["<|endoftext|>"]
#     # raw_str="low low low <|endoftext|> owowow aowowb lower lower lowest , ，"
#     # raw_str="low low"
#     vocab,word_bytes_freq,pair_freq=Tokenizer.__init_train_variables__(raw_str,special_tokens)
#     vocab,merge_list=Tokenizer.train_bpe(vocab,word_bytes_freq,pair_freq,500)
#     pass