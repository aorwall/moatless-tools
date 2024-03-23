**0 - /tmp/repos/scikit-learn/sklearn/ensemble/weight_boosting.py**:
```python
class BaseWeightBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):


    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format"""
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree, BaseForest))):
            X = check_array(X, accept_sparse='csr', dtype=DTYPE)

        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        return X
```
