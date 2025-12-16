from collections import defaultdict
from gc import collect
from typing import Iterable, Iterator
import json
import copy
import regex as re


class Tokenizer():
    
    PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def __init__(self, vocab:dict[int,bytes]={}, merges:list[tuple[bytes,bytes]]={}, special_tokens:list[str]=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens
        self.merges_rank_dict=self.__set_merges_rank__()
        self.reverse_vocab=self.__set_reverse_vocab__()
            
    def __set_merges_rank__(self):
        merges_rank_dict={}
        merges_len=len(self.merges)
        for i in range(merges_len):
            merges_rank_dict[self.merges[i]]=i
        return merges_rank_dict
    
    def __set_reverse_vocab__(self):
        return dict(zip(self.vocab.values(),self.vocab.keys()))
            
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
        with open(vocab_filepath,"r",encoding='utf-8') as vocab_file:
            vocab_str=vocab_file.read()
            vocab_load=json.loads(vocab_str)
            new_vocab=copy.deepcopy(vocab_load)
            for st in special_tokens:
                if st not in vocab_load.keys():
                    new_vocab[st]=len(new_vocab)
        with open(merges_filepath,"r",encoding='utf-8') as merges_file:
            new_merges=[
                tuple([
                    line_str.strip().split(' ')[0],
                    line_str.strip().split(' ')[1]
            ]) for line_str in merges_file.readlines()]
        return Tokenizer(new_vocab,new_merges,special_tokens)
    
    def split_by_special_tokens(cls, raw_str:str, special_tokens:list[str]):
        split_pat=r"("+"|".join(map(re.escape,special_tokens))+r")"
        parts = re.split(split_pat,raw_str)
        result = [re.findall(cls.PAT,part)  for part in parts if (part!='' and part not in special_tokens)]
        return result
    
    def flatten_to_bytes(self,items:Iterable):
        res=[]
        for item in items:
            if isinstance(item,Iterable):
                res.extend(item)
            else:
                res.append(item)
        return bytes(res)
    
    def pre_token_encode(self, text:str)->list[int]:
        byte_2_chr,chr_2_byte=Tokenizer.get_bytes_map()
        ## init
        word_bytes=tuple([byte_2_chr[x] for x in text.encode('utf-8')])
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
        result=[self.vocab.get(item) for item in word_bytes]
        return result
        
    
    def encode(self, text: str) -> list[int]:
        result=[]
        parts=self.split_by_special_tokens(text,self.special_tokens)
        for part in parts:
            for chunk in part:
                if chunk in self.vocab.keys():
                    result.append(self.vocab.get(chunk))
                else:
                    vocab_ids=self.pre_token_encode(chunk)
                    result.extend(vocab_ids)
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    
    def decode(self, ids: list[int]) -> str:
        ## encode: text(拆解)->byte->map_chr(聚合)->vocab_id
        ## decode: vocab_id->map_chr->byte(聚合)->text
        byte_2_chr,chr_2_byte=Tokenizer.get_bytes_map()
        vocab_chrs_list=[self.reverse_vocab.get(id) for id in ids]
        byte_list=[]
        for vocab_chrs in vocab_chrs_list:
            for vocab_chr in vocab_chrs:
                byte_list.append(chr_2_byte[vocab_chr])
        return bytes(byte_list).decode('utf-8',errors="replace")
    
vocab_filepath="/root/CS336/assignment1-basics/tests/fixtures/gpt2_vocab.json"
merges_filepath="/root/CS336/assignment1-basics/tests/fixtures/gpt2_merges.txt"
t=Tokenizer.from_files(vocab_filepath,merges_filepath,["<|endoftext|>"])
origin_text="I come from china，你呢"
encode_ids=t.encode(origin_text)
decode_str=t.decode(encode_ids)

print(origin_text)
print(encode_ids)
print(decode_str)


# print(t.split_by_special_tokens("ab cPAT1 defPAT2PAT3xyzPAT1PAT1 end    ",["PAT1", "PAT2", "PAT3"]))