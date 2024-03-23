**0 - /tmp/repos/scikit-learn/sklearn/linear_model/logistic.py**:
```python
if multi_class == 'ovr':
                scores = self.scores_[cls]
                coefs_paths = self.coefs_paths_[cls]
            else:
                # For multinomial, all scores are the same across classes
                scores = scores[0]
                # coefs_paths will keep its original shape because
                # logistic_regression_path expects it this way

            if self.refit:
                # best_index is between 0 and (n_Cs . n_l1_ratios - 1)
                # for example, with n_cs=2 and n_l1_ratios=3
                # the layout of scores is
                # [c1, c2, c1, c2, c1, c2]
                #   l1_1 ,  l1_2 ,  l1_3
                best_index = scores.sum(axis=0).argmax()

                best_index_C = best_index % len(self.Cs_)
                C_ = self.Cs_[best_index_C]
                self.C_.append(C_)

                best_index_l1 = best_index // len(self.Cs_)
                l1_ratio_ = l1_ratios_[best_index_l1]
                self.l1_ratio_.append(l1_ratio_)

                if multi_class == 'multinomial':
                    coef_init = np.mean(coefs_paths[:, :, best_index, :],
                                        axis=1)
                else:
                    coef_init = np.mean(coefs_paths[:, best_index, :], axis=0)

                # Note that y is label encoded and hence pos_class must be
                # the encoded label / None (for 'multinomial')
                w, _, _ = _logistic_regression_path(
                    X, y, pos_class=encoded_label, Cs=[C_], solver=solver,
                    fit_intercept=self.fit_intercept, coef=coef_init,
                    max_iter=self.max_iter, tol=self.tol,
                    penalty=self.penalty,
                    class_weight=class_weight,
                    multi_class=multi_class,
                    verbose=max(0, self.verbose - 1),
                    random_state=self.random_state,
                    check_input=False, max_squared_sum=max_squared_sum,
                    sample_weight=sample_weight,
                    l1_ratio=l1_ratio_)
                w = w[0]

            else:
                # Take the best scores across every fold and the average of
                # all coefficients corresponding to the best scores.
                best_indices = np.argmax(scores, axis=1)
                if multi_class == 'ovr':
                    w = np.mean([coefs_paths[i, best_indices[i], :]
                                 for i in range(len(folds))], axis=0)
                else:
                    w = np.mean([coefs_paths[:, i, best_indices[i], :]
                                 for i in range(len(folds))], axis=0)

                best_indices_C = best_indices % len(self.Cs_)
                self.C_.append(np.mean(self.Cs_[best_indices_C]))

                if self.penalty == 'elasticnet':
                    best_indices_l1 = best_indices // len(self.Cs_)
                    self.l1_ratio_.append(np.mean(l1_ratios_[best_indices_l1]))
                else:
                    self.l1_ratio_.append(None)

            
```
