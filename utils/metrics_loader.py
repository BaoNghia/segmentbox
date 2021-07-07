import numpy as np
from utils import metrics as custom_metrics


# You can use this file with your custom metrics
class Metrics:
    def __init__(self, metric_names):
        self.metric_names = metric_names
        # initialize a metric dictionary
        self.metric_dict = {metric_name: [] for metric_name in self.metric_names}

    def step(self, preds, labels):
        for metric in self.metric_names:
            # get the metric function
            do_metric = getattr(
                custom_metrics, metric, "The metric {} is not implemented".format(metric)
            )
            # check if metric require average method, if yes set to 'micro' or 'macro' or 'None'
            self.metric_dict[metric].append(do_metric(preds, labels, threshold = 0.5))

    def epoch(self):
        # calculate metrics for an entire epoch
        avg = [sum(metric) / (len(metric)) for metric in self.metric_dict.values()]
        metric_as_dict = dict(zip(self.metric_names, avg))
        self.metric_dict = {metric_name: [] for metric_name in self.metric_names}
        return metric_as_dict


    def last_step_metrics(self):
        # return metrics of last steps
        values = [self.metric_dict[metric][-1] for metric in self.metric_names]
        metric_as_dict = dict(zip(self.metric_names, values))
        self.metric_dict = {metric_name: [] for metric_name in self.metric_names}
        return metric_as_dict
