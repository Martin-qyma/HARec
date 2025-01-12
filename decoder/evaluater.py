import pickle
import evaluate
import numpy as np
from parse import args
from bert_score import BERTScorer

class MetricScore:
    def __init__(self):
        self.pred_input_path = f"../data/{args.dataset}/profile_pred.pkl"
        self.ref_input_path = f"../data/{args.dataset}/profile_ref.pkl"

        with open(self.pred_input_path, "rb") as f:
            self.data = pickle.load(f)
        with open(self.ref_input_path, "rb") as f:
            self.ref_data = pickle.load(f)

    def get_score(self):
        scores = {}
        (
            bert_precison,
            bert_recall,
            bert_f1,
            bert_precison_std,
            bert_recall_std,
            bert_f1_std,
        ) = BERT_score(self.data, self.ref_data)
        rouge = rouge_score(self.data, self.ref_data)
        bleu = bleu_score(self.data, self.ref_data)
        tokens_predict = [s.split() for s in self.data]
        usr, _ = unique_sentence_percent(tokens_predict)

        scores["blue_1"] = bleu["precisions"][0]
        scores["blue_4"] = bleu["precisions"][3]
        scores["rouge_1"] = float(rouge["rouge1"])
        scores["rouge_L"] = float(rouge["rougeL"])
        scores["bert_precision"] = bert_precison
        scores["bert_recall"] = bert_recall
        scores["bert_f1"] = bert_f1
        scores["usr"] = usr

        scores["bert_precision_std"] = bert_precison_std
        scores["bert_recall_std"] = bert_recall_std
        scores["bert_f1_std"] = bert_f1_std
        return scores

    def print_score(self):
        scores = self.get_score()
        print(f"dataset: {args.dataset}")
        print(f"model: {args.model}")
        print("Explanability Evaluation Metrics:")
        print(f"blue_1: {scores['blue_1']:.4f}")
        print(f"blue_4: {scores['blue_4']:.4f}")
        print(f"rouge_1: {scores['rouge_1']:.4f}")
        print(f"rouge_L: {scores['rouge_L']:.4f}")
        print(f"bert_precision: {scores['bert_precision']:.4f}")
        print(f"bert_recall: {scores['bert_recall']:.4f}")
        print(f"bert_f1: {scores['bert_f1']:.4f}")
        print(f"usr: {scores['usr']:.4f}")
        print("-"*30)
        print("Standard Deviation:")
        print(f"bert_precision_std: {scores['bert_precision_std']:.4f}")
        print(f"bert_recall_std: {scores['bert_recall_std']:.4f}")
        print(f"bert_f1_std: {scores['bert_f1_std']:.4f}")

def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for wa, wb in zip(sa, sb):
        if wa != wb:
            return False
    return True

def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        # seq is a list of words
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)

def BERT_score(predictions, references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        rescale_with_baseline=True,
    )
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]
    return (
        np.mean(precision),
        np.mean(recall),
        np.mean(f1),
        np.std(precision),
        np.std(recall),
        np.std(f1),
    )


def rouge_score(predictions, references):
    """
    predictions: list
    references: list or list[list]

    >>> rouge = evaluate.load('rouge')
    >>> predictions = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> results = rouge.compute(predictions=predictions,
    ...                         references=references)
    >>> print(results)
    {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return results


def bleu_score(predictions, references):
    """
    predictions: list of strs
    references: list of lists of strs

    >>> predictions = ["hello there general kenobi", "foo bar foobar"]
    >>> references = [
    ...     ["hello there general kenobi", "hello there !"],
    ...     ["foo bar foobar"]
    ... ]
    >>> bleu = evaluate.load("bleu")
    >>> results = bleu.compute(predictions=predictions, references=references)
    >>> print(results)
    {'bleu': 1.0, 'precisions': [1.0, 1.0, 1.0, 1.0], 'brevity_penalty': 1.0, 'length_ratio': 1.1666666666666667, 'translation_length': 7, 'reference_length': 6}
    """
    # change list into a list of lists
    bleu = evaluate.load("bleu")
    references = [[element] for element in references]
    results = bleu.compute(predictions=predictions, references=references)
    return results