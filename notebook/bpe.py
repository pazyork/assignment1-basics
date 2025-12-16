from collections import Counter
from typing import Iterable, Tuple
import json
import regex as re

def pretokenizer(raw_str:str,
                 special_tokens:list[str],
                 pat:str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):
    # 构建vocab
    vocab={i:chr(i) for i in range(256)}
    vocab.update({item[0]:item[1].encode('utf-8') for item in zip(range(256,256+len(special_tokens)),special_tokens)})
    # 替换掉special_tokens，并进行分词
    if special_tokens:
        chunks=re.split("|".join(map(re.escape,special_tokens)),raw_str)
    else:
        chunks=[raw_str]
    word_arr:list[str]=[item for chunk in chunks for item in re.findall(pat, chunk)]
    # 统计分词后词频
    word_bytes_freq = Counter([
        tuple(bytes([b]) for b in x.encode('utf-8'))
        for x in word_arr
    ])
    # 统计初始状态邻接词频
    pair_freq=Counter()
    for word_bytes in word_bytes_freq.keys():
        pair_freq.update(zip(word_bytes,word_bytes[1:]))
    return vocab,word_bytes_freq,pair_freq

def flatten_to_bytes(items:Iterable):
    res=[]
    for item in items:
        if isinstance(item,Iterable):
            res.extend(item)
        else:
            res.append(item)
    return bytes(res)

def update_pair_freq(word_bytes_freq:Counter,pair_freq:Counter,max_freq_pair:tuple):
    """
    作用：上一轮merge操作造成了对词频的影响，需要更新pair_freq
    时机：max_freq_pair对于word_bytes_freq的合并已经在更新pair_freq之前
    Args:
        max_freq_pair: (b'o',b'w')
        word_bytes_freq (Counter): [(b'l', b'ow'), (b' ', b'l', b'ow'),... ]
        pair_freq (Counter): [[(b' ', b'l'),(b'l', b'o'),(b'o', b'w')],....]--->[[(b' ', b'l'),(b'l', b'ow')],....]
    """
    max_freq_pair_bytes=flatten_to_bytes(max_freq_pair)
    # 首先保证当前word包含了max_freq_pair，这样才有可能需要做更行
    for word_bytes in filter(lambda x:max_freq_pair_bytes in x,word_bytes_freq.keys()):
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
            pair_freq[new_pair]=pair_freq[new_pair]+word_bytes_freq[word_bytes]
    return pair_freq

def merge_max_pair(word_bytes_freq:Counter,max_freq_pair:tuple,pair_freq:Counter):
    """根据max_freq_pair，更新原有的word_bytes分割、合并方式
    Args:
        max_freq_pair (tuple): (b'o',b'w')
        word_bytes_freq (Counter): [(b'l', b'o', b'w'), (b' ', b'l', b'o', b'w'),... ]-> [(b'l', b'ow'), (b' ', b'l', b'ow'),... ]
    """
    new_word_bytes_freq=word_bytes_freq.copy()
    max_freq_pair_bytes=flatten_to_bytes(max_freq_pair)
    for word_bytes in word_bytes_freq.keys():
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
        del new_word_bytes_freq[word_bytes]
        new_word_bytes_freq[tuple(new_word_bytes)]=word_bytes_freq[word_bytes]
    return new_word_bytes_freq
                
def train_bpe(vocab:dict,word_bytes_freq:Counter,pair_freq:Counter,vocab_max_limit:int):
    merge_list=[]
    max_freq_pair=None
    while(len(vocab)<=vocab_max_limit):
        if max_freq_pair:
            ## 由max_freq_pair的新合并现状，更新pair_freq
            pair_freq=update_pair_freq(word_bytes_freq,pair_freq,max_freq_pair)
        ## 找到pair_freq更新后[新的max_freq_pair]
        max_freq_pair=max(pair_freq,key=lambda x:[pair_freq[x],x])
        if pair_freq[max_freq_pair]==0:
            return vocab,merge_list
        max_freq_pair_bytes=flatten_to_bytes(max_freq_pair)
        print(len(vocab),max_freq_pair_bytes)
        vocab[len(vocab)]=max_freq_pair_bytes
        merge_list.append(max_freq_pair)
        ## 对word_bytes_freq中涉及[新的max_freq_pair]做合并操作
        word_bytes_freq=merge_max_pair(word_bytes_freq,max_freq_pair,pair_freq)
    return vocab,merge_list


with open('/root/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt') as f:
    raw_str=f.read()
    # raw_str="low low low <|endoftext|> owowow aowowb lower lower lowest , ，"
    # raw_str="low low"
    vocab,word_bytes_freq,pair_freq=pretokenizer(raw_str,['<|endoftext|>'])
    vocab,merge_list=train_bpe(vocab,word_bytes_freq,pair_freq,10000)
    # 保存vocab和merge_list到文件中
    def bytes_to_int_list(b:bytes):
        return list(b)
    def bytes_to_utf8(b):
        if isinstance(b,str):
            return b
        # 尝试UTF-8解码，失败时以�替换
        return b.decode('utf-8',errors='replace')
    vocab_serialized={str(k):bytes_to_int_list(v) for k,v in vocab.items()}
    vocab_utf8={str(k):bytes_to_utf8(v) for k,v in vocab.items()}
    merge_list_serialized=[[bytes_to_int_list(a),bytes_to_int_list(b)] for a,b in merge_list]
    with open('/root/CS336/assignment1-basics/data/vocab.json','w') as vf:
        json.dump(vocab_serialized,vf,ensure_ascii=False,indent=2)
    with open('/root/CS336/assignment1-basics/data/vocab_utf8.json','w') as vuf:
        json.dump(vocab_utf8,vuf,ensure_ascii=False,indent=2)
    with open('/root/CS336/assignment1-basics/data/merge_list.json','w') as mf:
        json.dump(merge_list_serialized,mf,ensure_ascii=False,indent=2)