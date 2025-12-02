import os
import json
import torch
import argparse
import numpy as np
import editdistance
import matplotlib.pyplot as plt
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='vocab/test_2.5w_data.pkl', type=str)
parser.add_argument('--vocab-file', default='vocab/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/Bob1/Rayleigh/resume/checkpoint_best.pth', type=str)
parser.add_argument('--channel', default='Rayleigh_MRC', type=str, help='Please choose AWGN, Rayleigh, Rician, or None')
parser.add_argument('--output-dir', default='results/Bob1/Rayleigh', type=str, help='Output directory for saving plots')
parser.add_argument('--output-name', default='performance_comparison_Rayleigh_resume_bestepochs_8_-10db.png', type=str)
parser.add_argument('--MAX-LENGTH', default=52, type=int)
parser.add_argument('--MIN-LENGTH', default=10, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=1, type = int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type = str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# using pre-trained model to compute the sentence similarity
# class Similarity():
#     def __init__(self, config_path, checkpoint_path, dict_path):
#         self.model1 = build_bert_model(config_path, checkpoint_path, with_pool=True)
#         self.model = keras.Model(inputs=self.model1.input,
#                                  outputs=self.model1.get_layer('Encoder-11-FeedForward-Norm').output)
#         # build tokenizer
#         self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
#
#     def compute_similarity(self, real, predicted):
#         token_ids1, segment_ids1 = [], []
#         token_ids2, segment_ids2 = [], []
#         score = []
#
#         for (sent1, sent2) in zip(real, predicted):
#             sent1 = remove_tags(sent1)
#             sent2 = remove_tags(sent2)
#
#             ids1, sids1 = self.tokenizer.encode(sent1)
#             ids2, sids2 = self.tokenizer.encode(sent2)
#
#             token_ids1.append(ids1)
#             token_ids2.append(ids2)
#             segment_ids1.append(sids1)
#             segment_ids2.append(sids2)
#
#         token_ids1 = keras.preprocessing.sequence.pad_sequences(token_ids1, maxlen=32, padding='post')
#         token_ids2 = keras.preprocessing.sequence.pad_sequences(token_ids2, maxlen=32, padding='post')
#
#         segment_ids1 = keras.preprocessing.sequence.pad_sequences(segment_ids1, maxlen=32, padding='post')
#         segment_ids2 = keras.preprocessing.sequence.pad_sequences(segment_ids2, maxlen=32, padding='post')
#
#         vector1 = self.model.predict([token_ids1, segment_ids1])
#         vector2 = self.model.predict([token_ids2, segment_ids2])
#
#         vector1 = np.sum(vector1, axis=1)
#         vector2 = np.sum(vector2, axis=1)
#
#         vector1 = normalize(vector1, axis=0, norm='max')
#         vector2 = normalize(vector2, axis=0, norm='max')
#
#         dot = np.diag(np.matmul(vector1, vector2.T))  # a*b
#         a = np.diag(np.matmul(vector1, vector1.T))  # a*a
#         b = np.diag(np.matmul(vector2, vector2.T))
#
#         a = np.sqrt(a)
#         b = np.sqrt(b)
#
#         output = dot / (a * b)
#         score = output.tolist()
#
#         return score

def compute_character_error_rate(sent1, sent2):
    sent1_clean = sent1.replace('<START>', '').replace(' ', '')
    sent2_clean = sent2.replace('<START>', '').replace(' ', '')
    
    distance = editdistance.eval(sent1_clean, sent2_clean)
    
    max_length = max(len(sent1_clean), len(sent2_clean))
    if max_length == 0:
        return 0.0
    return distance / max_length

def plot_results(SNR, all_bleu_results, all_cer_results, channel, output_dir='results'):
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
    plt.rcParams['axes.unicode_minus'] = False    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    for i, (vocab_name, bleu_scores) in enumerate(all_bleu_results.items()):
        ax1.plot(SNR, bleu_scores, color=colors[i], marker=markers[i], 
                linewidth=2, markersize=8, label=vocab_name)
    
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title(f'BLEU Score vs SNR ({channel} Channel)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.1) 
    
    for i, (vocab_name, cer_scores) in enumerate(all_cer_results.items()):
        ax2.plot(SNR, cer_scores, color=colors[i], marker=markers[i], 
                linewidth=2, markersize=8, label=vocab_name)
    
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Levenshtein Distance', fontsize=12)
    ax2.set_title(f'Levenshtein Distance vs SNR ({channel} Channel)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim(bottom=0)  

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, args.output_name)
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"\n图表已保存为: {png_path} ")


def performance(args, SNR, net, vocab_data, vocab_name, original_vocab_data=None):
    print(f"\n=== 词汇表: {vocab_name} ===")
    
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    test_eur = EurDataset('test_2.5w')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    token_to_idx = vocab_data['token_to_idx']
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    
    StoT = SeqtoText(token_to_idx, end_idx)
    
    if original_vocab_data is not None:
        original_token_to_idx = original_vocab_data['token_to_idx']
        original_pad_idx = original_token_to_idx["<PAD>"]
        original_start_idx = original_token_to_idx["<START>"]
        original_end_idx = original_token_to_idx["<END>"]
        original_StoT = SeqtoText(original_token_to_idx, original_end_idx)
    else:
        original_StoT = StoT
    
    score = []
    cer_score = []  
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []
            epoch_cer_scores = [] 

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:

                    sents = sents.to(device)
                    target = sents

                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(original_StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

                cer_list = []
                for sent1, sent2 in zip(word, target_word):
                    cer = compute_character_error_rate(sent1, sent2)
                    cer_list.append(cer)
                current_cer = np.mean(cer_list) if cer_list else 0.0
                epoch_cer_scores.append(current_cer)

            bleu_score = []
            for sent_list1, sent_list2 in zip(Tx_word, Rx_word):
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent_list1, sent_list2))
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)

            cer_score.append(epoch_cer_scores)

    score1 = np.mean(np.array(score), axis=0)
    cer_by_snr = np.mean(np.array(cer_score), axis=0)
    
    print("\n每个epoch的BLEU分数:")
    for i, epoch_score in enumerate(score):
        print(f"Epoch {i+1}: {epoch_score}")
    
    print(f"\n平均误字率（按SNR）:")
    for i, snr in enumerate(SNR):
        print(f"SNR {snr}dB: {cer_by_snr[i]:.6f}")
    
    print(f"\n平均BLEU分数: {score1}")

    return score1, cer_by_snr


if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12]
    #SNR = [-10,-9,-8,-7,-6]
    #SNR = [0,2,4,6,8,10,12,14,16,18,20]
    #SNR = [-4,-2,0,2,4,6,8,10,12]
    # 定义三个词汇表文件
    vocab_files = [
        ('vocab.json', 'Bob1'),
        ('vocab_random_1.json', 'Bob2'),
        ('vocab_random_2.json', 'Bob3')
    ]

    # 加载模型
    vocab = json.load(open('vocab/vocab.json', 'r', encoding='utf-8'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    checkpoint = torch.load(args.checkpoint_path)
    state_dict = checkpoint['state_dict']
    deepsc.load_state_dict(state_dict)
    print('模型加载完成!')

    original_vocab_data = json.load(open('vocab/vocab.json', 'r', encoding='utf-8'))
    
    all_bleu_results = {}
    all_cer_results = {}
    
    for vocab_file, vocab_name in vocab_files:
        vocab_path = f'vocab/{vocab_file}'
        if os.path.exists(vocab_path):
            print(f"\n正在使用词汇表: {vocab_file}")
            vocab_data = json.load(open(vocab_path, 'r', encoding='utf-8'))
            bleu_score, cer_score = performance(args, SNR, deepsc, vocab_data, vocab_name, original_vocab_data)
            all_bleu_results[vocab_name] = bleu_score
            all_cer_results[vocab_name] = cer_score
        else:
            print(f"词汇表文件不存在: {vocab_path}")

    print("\n" + "="*50)
    print("所有词汇表性能比较:")
    print("="*50)
    print("\nBLEU分数比较:")
    for vocab_name, score in all_bleu_results.items():
        print(f"{vocab_name}: {score}")
    
    print("\n误字率比较（按SNR）:")
    for vocab_name, cer_scores in all_cer_results.items():
        print(f"{vocab_name}:")
        for i, snr in enumerate(SNR):
            print(f"  SNR {snr}dB: {cer_scores[i]:.6f}")

    print("\n" + "="*50)
    plot_results(SNR, all_bleu_results, all_cer_results, args.channel, args.output_dir)

    #similarity.compute_similarity(sent1, real)
    #similarity.compute_similarity(sent1, real)
