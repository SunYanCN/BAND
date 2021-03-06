import numpy as np
from tensorflow.keras.callbacks import Callback
from band.seqeval.metrics import f1_score, classification_report


class F1Metrics(Callback):

    def __init__(self, id2label, pad_value=0, validation_data=None, steps=None, digits=4):
        """
        Args:
            id2label (dict): id to label mapping.
            (e.g. {1: 'B-LOC', 2: 'I-LOC'})
            pad_value (int): padding value.
            digits (int or None): number of digits in printed classification report
              (use None to print only F1 score without a report).
        """
        super(F1Metrics, self).__init__()
        if type(id2label) == list:
            self.id2label = {i: label for i, label in enumerate(id2label)}
        else:
            self.id2label = id2label
        self.pad_value = pad_value
        self.validation_data = validation_data
        self.steps = steps
        self.digits = digits
        self.is_fit = validation_data is None

    def convert_idx_to_name(self, y, array_indexes):
        """Convert label index to name.

        Args:
            y (np.ndarray): label index 2d array.
            array_indexes (list): list of valid index arrays for each row.

        Returns:
            y: label name list.
        """
        y = [[self.id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    def predict(self, dataset, steps):
        """Predict sequences.
        Args:
            X (np.ndarray): input data.
            y (np.ndarray): tags.

        Returns:
            y_true: true sequences.
            y_pred: predicted sequences.
        """
        y_pred = self.model.predict(dataset, steps=steps)
        y_true = np.concatenate([y.numpy() for _, y in dataset], axis=0)

        # reduce dimension.
        y_pred = np.argmax(y_pred, -1)
        non_pad_indexes = [np.nonzero(y_true_row != self.pad_value)[0] for y_true_row in y_true]

        y_true = self.convert_idx_to_name(y_true, non_pad_indexes)
        y_pred = self.convert_idx_to_name(y_pred, non_pad_indexes)

        return y_true, y_pred

    def score(self, y_true, y_pred):
        """Calculate f1 score.

        Args:
            y_true (list): true sequences.
            y_pred (list): predicted sequences.

        Returns:
            score: f1 score.
        """
        score = f1_score(y_true, y_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        if self.digits:
            print(classification_report(y_true, y_pred, digits=self.digits))
        return score

    def on_epoch_end(self, epoch, logs={}):
        if self.is_fit:
            self.on_epoch_end_fit(epoch, logs)
        else:
            self.on_epoch_end_fit_generator(epoch, logs)

    def on_epoch_end_fit(self, epoch, logs={}):
        y_true, y_pred = self.predict(self.validation_data, steps=self.steps)
        score = self.score(y_true, y_pred)
        logs['f1'] = score

    def on_epoch_end_fit_generator(self, epoch, logs={}):
        y_true, y_pred = self.predict(self.validation_data, steps=self.steps)
        score = self.score(y_true, y_pred)
        logs['f1'] = score
