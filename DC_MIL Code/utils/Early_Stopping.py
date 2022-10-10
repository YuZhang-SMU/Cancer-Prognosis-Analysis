import numpy as np
import torch

class EarlyStopping:
    def __init__(self, model_path, patience=5, verbose=False):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf
        self.model_path = model_path

    def __call__(self, val_metric, model):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        if self.verbose:
            print('Validation metric improved ({} --> {}).  Saving model ...{}'.format(self.val_metric_min, val_metric, self.model_path))
        torch.save(model.state_dict(), self.model_path)
        self.val_metric_min = val_metric