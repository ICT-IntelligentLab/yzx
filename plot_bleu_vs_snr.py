import matplotlib.pyplot as plt
import numpy as np

def plot_bleu_vs_snr():
    """绘制BLEU分数随SNR变化的图表"""
    
    SNR = [0, 3, 6, 9, 12, 15, 18]
    bleu_scores = [0.19780757, 0.46520884, 0.7570006, 0.89004894, 0.93634328, 0.95074933, 0.95584709]
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] 
    plt.rcParams['axes.unicode_minus'] = False 
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(SNR, bleu_scores, color='blue', marker='o', 
            linewidth=2, markersize=8, label='BLEU Score')
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('BLEU Score', fontsize=12)
    plt.title('BLEU Score vs SNR', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(0, 1.1)  

    plt.tight_layout()
    plt.savefig('bleu_vs_snr.png', dpi=300, bbox_inches='tight')
    plt.savefig('bleu_vs_snr.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"\n图表已保存为: bleu_vs_snr.png 和 bleu_vs_snr.pdf")
    print(f"SNR值: {SNR}")
    print(f"BLEU分数: {bleu_scores}")

if __name__ == "__main__":
    plot_bleu_vs_snr()
