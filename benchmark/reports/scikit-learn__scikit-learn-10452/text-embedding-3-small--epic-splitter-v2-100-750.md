**0 - /tmp/repos/scikit-learn/sklearn/preprocessing/data.py**:
```python
class PolynomialFeatures(BaseEstimator, TransformerMixin):


    def transform(self, X):
        """Transform data to polynomial features

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_features]
            The data to transform, row by row.
            Sparse input should preferably be in CSC format.

        Returns
        -------
        XP : np.ndarray or CSC sparse matrix, shape [n_samples, NP]
            The matrix of features, where NP is the number of polynomial
            features generated from the combination of inputs.
        """
        check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = check_array(X, dtype=FLOAT_DTYPES, accept_sparse='csc')
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        combinations = self._combinations(n_features, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        if sparse.isspmatrix(X):
            columns = []
            for comb in combinations:
                if comb:
                    out_col = 1
                    for col_idx in comb:
                        out_col = X[:, col_idx].multiply(out_col)
                    columns.append(out_col)
                else:
                    columns.append(sparse.csc_matrix(np.ones((X.shape[0], 1))))
            XP = sparse.hstack(columns, dtype=X.dtype).tocsc()
        else:
            XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
            for i, comb in enumerate(combinations):
                XP[:, i] = X[:, comb].prod(1)

        return XP
```
