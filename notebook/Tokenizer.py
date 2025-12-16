from gc import collect
from typing import Iterable, Iterator
import json
import copy
import regex as re
from assignment1-basics.implement.Tokenizer import Tokenizer


class Tokenizer():
    def __init__(self, vocab:dict[int,bytes]={}, merges:list[tuple[bytes,bytes]]={}, special_tokens:list[str]=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens
    
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens:list[str]=None):
        new_vocab={}
        new_merges=[]
        with open(vocab_filepath,"r",encoding='utf-8') as vocab_file:
            vocab_str=vocab_file.read()
            vocab_load=json.load(vocab_str)
            new_vocab=copy.deepcopy(vocab_load)
            for st in special_tokens:
                if st not in vocab_file.keys():
                    new_vocab[st]=len(new_vocab)
        with open(merges_filepath,"r",encoding='utf-8') as merges_file:
            new_merges=[
                tuple(
                    line_str.split(' ')[0].encode('utf-8'),
                    line_str.split(' ')[1].encode('utf-8')
                ) for line_str in merges_file.readlines()]
        return Tokenizer(new_vocab,new_merges,special_tokens)
    
    def split_by_special_tokens(cls, raw_str:str, special_tokens:list[str]):
        split_pat="("+"|".join(map(re.escape,special_tokens))+f")"
        parts = re.split(split_pat,raw_str)
        return [part for part in parts if part!='']
    
    def flatten_to_bytes(self,items:Iterable):
        res=[]
        for item in items:
            if isinstance(item,Iterable):
                res.extend(item)
            else:
                res.append(item)
        return bytes(res)
    
    def pre_token_encode(self, text:str)->list[int]:
        ## init
        word_bytes=tuple([bytes([x]) for x in text.encode('utf-8')])
        merges_len=len(self.merges)
        ## 循环，找到最大的
        while True:
            # 统计当前pair对中，优先级最高的
            merge_idx_rank={}
            curr_word_bytes_len=len(word_bytes.keys())
            for i in range(curr_word_bytes_len-1):
                for j in range(merges_len):
                    if word_bytes[i]==self.merges[j][0] and word_bytes[i+1]==self.merges[j][1]:
                        merge_idx_rank[i]=j
            min_idx=min(merge_idx_rank.keys(),key=lambda x:merge_idx_rank[x])
            # 处理要合并的pair
            if not (min_idx):
                break
            new_word_bytes=tuple()
            i=0
            while i < curr_word_bytes_len:
                if i == min_idx:
                    new_word_bytes+=(word_bytes[i]+word_bytes[i+1])
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
        parts=self.split_by_special_tokens(text,self.merges)
        for part in parts:
            if part in self.vocab.keys():
                result.append(self.vocab.get(part))
            else:
                pass
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    
    def decode(self, ids: list[int]) -> str:
        pass
    
t=Tokenizer.from_files()
Tokenizer.f

# print(t.split_by_special_tokens("ab cPAT1 defPAT2PAT3xyzPAT1PAT1 end    ",["PAT1", "PAT2", "PAT3"]))
t.pre_token_encode("ab cPAT1 defPAT2PAT3xyzPAT1PAT1 end    ")