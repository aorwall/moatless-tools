**0 - /tmp/repos/scikit-learn/sklearn/feature_extraction/text.py**:
```python
class TfidfVectorizer(CountVectorizer):


    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
                          "be converted to np.float64."
                          .format(FLOAT_DTYPES, self.dtype),
                          UserWarning)
```
