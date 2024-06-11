from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
import numpy as np


def recall_at_5_fpr(y_true, y_probs):
    '''Calculate the recall at 5% false positive rate'''
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    idx = np.argmax(fpr >= 0.05)
    return tpr[idx]

metric_name2func = {
    'tp': lambda y_true, y_probs: sum((y_true == 1) & (y_probs >= 0.5)),
    'tn': lambda y_true, y_probs: sum((y_true == 0) & (y_probs < 0.5)),
    'fp': lambda y_true, y_probs: sum((y_true == 0) & (y_probs >= 0.5)),
    'fn': lambda y_true, y_probs: sum((y_true == 1) & (y_probs < 0.5)),
    'auc': roc_auc_score,
    'aucpr': average_precision_score,
    'rec@5fpr': recall_at_5_fpr,
}

class MetricStore():
    def __init__(self,
                 metric_names=['loss', 'tp', 'tn', 'fp', 'fn', 'auc', 'aucpr', 'rec@5fpr'],
                 sets=['training', 'validation']):
        
        self.metric_names = metric_names
        self.sets = sets
        self.metrics = {metric_name: {set_: [] for set_ in sets} for metric_name in metric_names}

    def save(self, saver):
        '''Save the metrics to the slune saver object'''
        for metric_name in self.metric_names:
            for set_ in self.sets:
                for i in range(len(self.metrics[metric_name][set_])):
                    saver.log({f'{set_}_{metric_name}': self.metrics[metric_name][set_][i]})
        saver.save_collated()
    
    def log(self, metrics):
        '''Log the metrics to the metric store'''
        for metric_name, sets in metrics.items():
            for set_, value in sets.items():
                value_ = value.item() if hasattr(value, 'item') else value
                self.metrics[metric_name][set_].append(value_)

    def calculate_metrics(self, y_true, y_probs, set_name):
        '''Calculate the metrics for the given data'''
        metrics = {metric_name: {} for metric_name in self.metric_names}
        for metric_name in self.metric_names:
            if metric_name == 'loss':
                continue
            metrics[metric_name][set_name] = metric_name2func[metric_name](y_true, y_probs)

        self.log(metrics)
        return metrics
    

