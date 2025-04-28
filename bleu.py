import nltk
import re
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def compute_bleu(pred_caption, references):
    """
    pred_caption: str, predicted caption
    references: list of str, ground truth captions
    """
    # 1. Tokenize
    pred_tokens = re.sub(r'[^\w\s]', '', pred_caption).lower().split()
    ref_tokens =  [re.sub(r'[^\w\s]', '', ref).lower().split() for ref in references]
    pred_tokens = [pred_tokens]
    smoothie = SmoothingFunction().method1

    # 2. Compute BLEU-4
    bleu_score = corpus_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),  smoothing_function=smoothie)

    return bleu_score


def main():
    # Example
    predicted = "the weather is good"
    references = ["the sky is clear", "the weather is extremely good"]

    score = compute_bleu(predicted, references)
    print(f"BLEU Score: {score:.4f}")

if __name__ == "__main__":
    main()