'''
[1 2 3 4 5 6 7 8 9]
[0 0 0 0 0 1 1 1 1]

[
    [[1 2 3 4 5 6], [7 8 9]],
    [[4 5 6 7 8 9], [1 2 3]],
    [[7 8 9 1 2 3], [4 5 6]]
]
[
    [[0 0 0 0 0 1], [1 1 1]]
]
'''

from sklearn.model_selection import ShuffleSplit

class CrossValidationSplitter:
    def __init__(self):
        self.data = []

    def do_cross_validation(self, data, labels, folds=3):
        cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
        print cv