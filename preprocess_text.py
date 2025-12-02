import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import os
import json
from tqdm import tqdm
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input-data-dir', default='wikipedia', type=str, help='输入数据目录，包含parquet文件')
parser.add_argument('--output-train-dir', default='vocab/train_data.pkl', type=str)
parser.add_argument('--output-test-dir', default='vocab/test_data.pkl', type=str)
parser.add_argument('--output-vocab', default='vocab/vocab.json', type=str)
parser.add_argument('--output-vocab-sorted', default='vocab/vocab_random1.json', type=str)
parser.add_argument('--output-vocab-random', default='vocab/vocab_random2.json', type=str)

SPECIAL_TOKENS = {
  '<PAD>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}

def normalize_chinese_string(s):
    """
    中文文本标准化处理
    """
    s = remove_tags(s)
    s = re.sub(r'[^\u4e00-\u9fa5，。！？；：、\s]', '', s)
    s = re.sub(r'\s*([，。！？；：、])\s*', r'\1', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

def cutted_data(cleaned, MIN_LENGTH=10, MAX_LENGTH=50):
    """
    中文文本切割，按字符而不是单词
    """
    cutted_lines = list()
    for line in cleaned:
        length = len(line.strip())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            cutted_lines.append(line.strip())
    return cutted_lines

def save_clean_sentences(sentence, save_path):
    pickle.dump(sentence, open(save_path, 'wb'))
    print('Saved: %s' % save_path)

def process_local_wikipedia_dataset(data_dir):
    """
    处理本地下载的维基百科数据集
    """
    print(f"Loading local dataset from: {data_dir}")
    
    sentences = []
    
    # 读取所有parquet文件
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if not parquet_files:
        raise ValueError(f"在目录 {data_dir} 中没有找到parquet文件")
    
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    for parquet_file in parquet_files:
        file_path = os.path.join(data_dir, parquet_file)
        print(f"处理文件: {parquet_file}")
        
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = row['text']
            if pd.notna(text) and text.strip():
                paragraphs = text.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        normalized = normalize_chinese_string(paragraph)
                        if normalized:
                            sentences.append(normalized)
    
    return sentences


def tokenize(s, delim=' ',  add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    if delim == '':
        tokens = list(s)
    else:
        tokens = s.split(delim)
        
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, token_to_idx = { }, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None, ):
    token_to_count = {}

    for seq in sequences:
      seq_tokens = tokenize(seq, delim=delim, punct_to_keep=punct_to_keep,
                      punct_to_remove=punct_to_remove,
                      add_start_token=False, add_end_token=False)
      for token in seq_tokens:
        if token not in token_to_count:
          token_to_count[token] = 0
        token_to_count[token] += 1

    for token, count in sorted(token_to_count.items()):
      if count >= min_token_count:
        token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
      if token not in token_to_idx:
        if allow_unk:
          token = '<UNK>'
        else:
          raise KeyError('Token "%s" not in vocab' % token)
      seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
      tokens.append(idx_to_token[idx])
      if stop_at_end and tokens[-1] == '<END>':
        break
    if delim is None:
      return tokens
    else:
      return delim.join(tokens)


def main(args):
    print(f"Processing local dataset from: {args.input_data_dir}")
    
    sentences = process_local_wikipedia_dataset(args.input_data_dir)
    unique_sentences = list(set(sentences))
    print('Number of sentences: {}'.format(len(unique_sentences)))
    filtered_sentences = cutted_data(unique_sentences, MIN_LENGTH=10, MAX_LENGTH=50)
    print('Number of filtered sentences: {}'.format(len(filtered_sentences)))
    
    print('Build Vocab')
    token_to_idx = build_vocab(
        filtered_sentences, SPECIAL_TOKENS,
        delim='',  
        punct_to_keep=['，', '。', '！', '？', '；', '：', '、'],
        punct_to_remove=[]
    )

    vocab = {'token_to_idx': token_to_idx}
    print('Number of characters in Vocab: {}'.format(len(token_to_idx)))

    if args.output_vocab != '':
        os.makedirs(os.path.dirname(args.output_vocab), exist_ok=True)
        with open(args.output_vocab, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)

    random_tokens1 = list(token_to_idx.keys())
    random.shuffle(random_tokens1)
    random_token_to_idx1 = {}
    for idx, token in enumerate(random_tokens1):
        random_token_to_idx1[token] = idx
    
    random_vocab1 = {'token_to_idx': random_token_to_idx1}
    if args.output_vocab_sorted != '':
        os.makedirs(os.path.dirname(args.output_vocab_sorted), exist_ok=True)
        with open(args.output_vocab_sorted, 'w', encoding='utf-8') as f:
            json.dump(random_vocab1, f, ensure_ascii=False)

    random_tokens = list(token_to_idx.keys())
    random.shuffle(random_tokens)
    random_token_to_idx = {}
    for idx, token in enumerate(random_tokens):
        random_token_to_idx[token] = idx
    
    random_vocab = {'token_to_idx': random_token_to_idx}
    if args.output_vocab_random != '':
        os.makedirs(os.path.dirname(args.output_vocab_random), exist_ok=True)
        with open(args.output_vocab_random, 'w', encoding='utf-8') as f:
            json.dump(random_vocab, f, ensure_ascii=False)

    print('Start encoding text')
    results = []
    for seq in tqdm(filtered_sentences):
        words = tokenize(seq, delim='', 
                        punct_to_keep=['，', '。', '！', '？', '；', '：', '、'],
                        punct_to_remove=[])
        tokens = [token_to_idx[word] for word in words]
        results.append(tokens)

    print('Writing Data')
    os.makedirs(os.path.dirname(args.output_train_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_test_dir), exist_ok=True)
    
    train_data = results[: round(len(results) * 0.9)]
    test_data = results[round(len(results) * 0.9):]

    with open(args.output_train_dir, 'wb') as f:
        pickle.dump(train_data, f)
    with open(args.output_test_dir, 'wb') as f:
        pickle.dump(test_data, f)
        
    print(f"Training data saved to: {args.output_train_dir}")
    print(f"Test data saved to: {args.output_test_dir}")
    print(f"Original vocabulary saved to: {args.output_vocab}")
    print(f"First random vocabulary saved to: {args.output_vocab_sorted}")
    print(f"Second random vocabulary saved to: {args.output_vocab_random}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
