import numpy as np
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
