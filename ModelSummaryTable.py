from prettytable import PrettyTable

class ModelSummaryTable(object):
    def __init__(self):
        self.table = PrettyTable()
        self.table.field_names = ["Algorithm", "Test Accuracy", "Test F1-score", "Max_depth", "n_estimators"]

    def add_row(self, algo, accuracy, f1_score, max_depth=None, n_est=None):
        self.table.add_row([algo, accuracy, f1_score, max_depth, n_est])

    def show(self):
        print(self.table)