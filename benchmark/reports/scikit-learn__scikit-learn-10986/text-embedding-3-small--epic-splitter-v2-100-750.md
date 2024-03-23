**0 - /tmp/repos/scikit-learn/sklearn/linear_model/logistic.py**:
```python
if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == 'ovr':
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    'Initialization coef is of shape %d, expected shape '
                    '%d or %d' % (coef.size, n_features, w0.size))
            w0[:coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if (coef.shape[0] != n_classes or
                    coef.shape[1] not in (n_features, n_features + 1)):
                raise ValueError(
                    'Initialization coef is of shape (%d, %d), expected '
                    'shape (%d, %d) or (%d, %d)' % (
                        coef.shape[0], coef.shape[1], classes.size,
                        n_features, classes.size, n_features + 1))
            w0[:, :coef.shape[1]] = coef

    if multi_class == 'multinomial':
        # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
        if solver in ['lbfgs', 'newton-cg']:
            w0 = w0.ravel()
        target = Y_multi
        if solver == 'lbfgs':
            func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
        elif solver == 'newton-cg':
            func = lambda x, *args: _multinomial_loss(x, *args)[0]
            grad = lambda x, *args: _multinomial_loss_grad(x, *args)[1]
            hess = _multinomial_grad_hess
        warm_start_sag = {'coef': w0.T}
    else:
        target = y_bin
        if solver == 'lbfgs':
            func = _logistic_loss_and_grad
        elif solver == 'newton-cg':
            func = _logistic_loss
            grad = lambda x, *args: _logistic_loss_and_grad(x, *args)[1]
            hess = _logistic_grad_hess
        warm_start_sag = {'coef': np.expand_dims(w0, axis=1)}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    
```
