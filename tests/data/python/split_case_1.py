def lars_path(X, y, Xy=None, Gram=None, max_iter=500,
              alpha_min=0, method='lar', copy_X=True,
              eps=np.finfo(np.float).eps,
              copy_Gram=True, verbose=0, return_path=True,
              return_n_iter=False, positive=False):
    """Compute Least Angle Regression or Lasso path using LARS algorithm [1]

    The optimization objective for the case method='lasso' is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    in the case of method='lars', the objective function is only known in
    the form of an implicit equation (see discussion in [1])

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    -----------
    X : array, shape: (n_samples, n_features)
        Input data.

    y : array, shape: (n_samples)
        Input targets.

    Xy : array-like, shape (n_samples,) or (n_samples, n_targets), \
            optional
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    Gram : None, 'auto', array, shape: (n_features, n_features), optional
        Precomputed Gram matrix (X' * X), if ``'auto'``, the Gram
        matrix is precomputed from the given X, if there are more samples
        than features.

    max_iter : integer, optional (default=500)
        Maximum number of iterations to perform, set to infinity for no limit.

    alpha_min : float, optional (default=0)
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.

    method : {'lar', 'lasso'}, optional (default='lar')
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    copy_X : bool, optional (default=True)
        If ``False``, ``X`` is overwritten.

    eps : float, optional (default=``np.finfo(np.float).eps``)
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    copy_Gram : bool, optional (default=True)
        If ``False``, ``Gram`` is overwritten.

    verbose : int (default=0)
        Controls output verbosity.

    return_path : bool, optional (default=True)
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, optional (default=False)
        Whether to return the number of iterations.

    positive : boolean (default=False)
        Restrict coefficients to be >= 0.
        This option is only allowed with method 'lasso'. Note that the model
        coefficients will not converge to the ordinary-least-squares solution
        for small values of alpha. Only coefficients up to the smallest alpha
        value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
        the stepwise Lars-Lasso algorithm are typically in congruence with the
        solution of the coordinate descent lasso_path function.

    Returns
    --------
    alphas : array, shape: [n_alphas + 1]
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller.

    active : array, shape [n_alphas]
        Indices of active variables at the end of the path.

    coefs : array, shape (n_features, n_alphas + 1)
        Coefficients along the path

    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set
        to True.

    See also
    --------
    lasso_path
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode

    References
    ----------
    .. [1] "Least Angle Regression", Effron et al.
           https://statweb.stanford.edu/~tibs/ftp/lars.pdf

    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_

    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_

    """
    if method == 'lar' and positive:
        warnings.warn('positive option is broken for Least'
                      ' Angle Regression (LAR). Use method="lasso".'
                      ' This option will be removed in version 0.22.',
                      DeprecationWarning)

    n_features = X.shape[1]
    n_samples = y.size
    max_features = min(max_iter, n_features)

    if return_path:
        coefs = np.zeros((max_features + 1, n_features))
        alphas = np.zeros(max_features + 1)
    else:
        coef, prev_coef = np.zeros(n_features), np.zeros(n_features)
        alpha, prev_alpha = np.array([0.]), np.array([0.])  # better ideas?

    n_iter, n_active = 0, 0
    active, indices = list(), np.arange(n_features)
    # holds the sign of covariance
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False

    # will hold the cholesky factorization. Only lower part is
    # referenced.
    L = np.empty((max_features, max_features), dtype=X.dtype)
    swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (X,))
    solve_cholesky, = get_lapack_funcs(('potrs',), (X,))

    if Gram is None or Gram is False:
        Gram = None
        if copy_X:
            # force copy. setting the array to be fortran-ordered
            # speeds up the calculation of the (partial) Gram matrix
            # and allows to easily swap columns
            X = X.copy('F')

    elif isinstance(Gram, str) and Gram == 'auto' or Gram is True:
        if Gram is True or X.shape[0] > X.shape[1]:
            Gram = np.dot(X.T, X)
        else:
            Gram = None
    elif copy_Gram:
        Gram = Gram.copy()

    if Xy is None:
        Cov = np.dot(X.T, y)
    else:
        Cov = Xy.copy()

    if verbose:
        if verbose > 1:
            print("Step\t\tAdded\t\tDropped\t\tActive set size\t\tC")
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    tiny32 = np.finfo(np.float32).tiny  # to avoid division by 0 warning
    equality_tolerance = np.finfo(np.float32).eps

    while True:
        if Cov.size:
            if positive:
                C_idx = np.argmax(Cov)
            else:
                C_idx = np.argmax(np.abs(Cov))

            C_ = Cov[C_idx]

            if positive:
                C = C_
            else:
                C = np.fabs(C_)
        else:
            C = 0.

        if return_path:
            alpha = alphas[n_iter, np.newaxis]
            coef = coefs[n_iter]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
            prev_coef = coefs[n_iter - 1]

        alpha[0] = C / n_samples
        if alpha[0] <= alpha_min + equality_tolerance:  # early stopping
            if abs(alpha[0] - alpha_min) > equality_tolerance:
                # interpolation factor 0 <= ss < 1
                if n_iter > 0:
                    # In the first iteration, all alphas are zero, the formula
                    # below would make ss a NaN
                    ss = ((prev_alpha[0] - alpha_min) /
                          (prev_alpha[0] - alpha[0]))
                    coef[:] = prev_coef + ss * (coef - prev_coef)
                alpha[0] = alpha_min
            if return_path:
                coefs[n_iter] = coef
            break

        if n_iter >= max_iter or n_active >= n_features:
            break

        if not drop:

            ##########################################################
            # Append x_j to the Cholesky factorization of (Xa * Xa') #
            #                                                        #
            #            ( L   0 )                                   #
            #     L  ->  (       )  , where L * w = Xa' x_j          #
            #            ( w   z )    and z = ||x_j||                #
            #                                                        #
            ##########################################################

            if positive:
                sign_active[n_active] = np.ones_like(C_)
            else:
                sign_active[n_active] = np.sign(C_)
            m, n = n_active, C_idx + n_active

            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = indices[m], indices[n]
            Cov_not_shortened = Cov
            Cov = Cov[1:]  # remove Cov[0]

            if Gram is None:
                X.T[n], X.T[m] = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active]) ** 2
                L[n_active, :n_active] = \
                    np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                # swap does only work inplace if matrix is fortran
                # contiguous ...
                Gram[m], Gram[n] = swap(Gram[m], Gram[n])
                Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]

            # Update the cholesky decomposition for the Gram matrix
            if n_active:
                linalg.solve_triangular(L[:n_active, :n_active],
                                        L[n_active, :n_active],
                                        trans=0, lower=1,
                                        overwrite_b=True,
                                        **solve_triangular_args)

            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag

            if diag < 1e-7:
                # The system is becoming too ill-conditioned.
                # We have degenerate vectors in our active set.
                # We'll 'drop for good' the last regressor added.

                # Note: this case is very rare. It is no longer triggered by
                # the test suite. The `equality_tolerance` margin added in 0.16
                # to get early stopping to work consistently on all versions of
                # Python including 32 bit Python under Windows seems to make it
                # very difficult to trigger the 'drop for good' strategy.
                warnings.warn('Regressors in active set degenerate. '
                              'Dropping a regressor, after %i iterations, '
                              'i.e. alpha=%.3e, '
                              'with an active set of %i regressors, and '
                              'the smallest cholesky pivot element being %.3e.'
                              ' Reduce max_iter or increase eps parameters.'
                              % (n_iter, alpha, n_active, diag),
                              ConvergenceWarning)

                # XXX: need to figure a 'drop for good' way
                Cov = Cov_not_shortened
                Cov[0] = 0
                Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
                continue

            active.append(indices[n_active])
            n_active += 1

            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, active[-1], '',
                                                      n_active, C))

        if method == 'lasso' and n_iter > 0 and prev_alpha[0] < alpha[0]:
            # alpha is increasing. This is because the updates of Cov are
            # bringing in too much numerical error that is greater than
            # than the remaining correlation with the
            # regressors. Time to bail out
            warnings.warn('Early stopping the lars path, as the residues '
                          'are small and the current value of alpha is no '
                          'longer well controlled. %i iterations, alpha=%.3e, '
                          'previous alpha=%.3e, with an active set of %i '
                          'regressors.'
                          % (n_iter, alpha, prev_alpha, n_active),
                          ConvergenceWarning)
            break

        # least squares solution
        least_squares, info = solve_cholesky(L[:n_active, :n_active],
                                             sign_active[:n_active],
                                             lower=True)

        if least_squares.size == 1 and least_squares == 0:
            # This happens because sign_active[:n_active] = 0
            least_squares[...] = 1
            AA = 1.
        else:
            # is this really needed ?
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))

            if not np.isfinite(AA):
                # L is too ill-conditioned
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += (2 ** i) * eps
                    least_squares, info = solve_cholesky(
                        L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]),
                              eps)
                    AA = 1. / np.sqrt(tmp)
                    i += 1
            least_squares *= AA

        if Gram is None:
            # equiangular direction of variables in the active set
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            # correlation between each unactive variables and
            # eqiangular vector
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            # if huge number of features, this takes 50% of time, I
            # think could be avoided if we just update it using an
            # orthogonal (QR) decomposition of X
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)

        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny32))
        if positive:
            gamma_ = min(g1, C / AA)
        else:
            g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny32))
            gamma_ = min(g1, g2, C / AA)

        # TODO: better names for these variables: z
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            # some coefficients have changed sign
            idx = np.where(z == z_pos)[0][::-1]

            # update the sign, important for LAR
            sign_active[idx] = -sign_active[idx]

            if method == 'lasso':
                gamma_ = z_pos
            drop = True

        n_iter += 1

        if return_path:
            if n_iter >= coefs.shape[0]:
                del coef, alpha, prev_alpha, prev_coef
                # resize the coefs and alphas array
                add_features = 2 * max(1, (max_features - n_active))
                coefs = np.resize(coefs, (n_iter + add_features, n_features))
                coefs[-add_features:] = 0
                alphas = np.resize(alphas, n_iter + add_features)
                alphas[-add_features:] = 0
            coef = coefs[n_iter]
            prev_coef = coefs[n_iter - 1]
        else:
            # mimic the effect of incrementing n_iter on the array references
            prev_coef = coef
            prev_alpha[0] = alpha[0]
            coef = np.zeros_like(coef)

        coef[active] = prev_coef[active] + gamma_ * least_squares

        # update correlations
        Cov -= gamma_ * corr_eq_dir

        # See if any coefficient has changed sign
        if drop and method == 'lasso':

            # handle the case when idx is not length of 1
            [arrayfuncs.cholesky_delete(L[:n_active, :n_active], ii) for ii in
                idx]

            n_active -= 1
            # handle the case when idx is not length of 1
            drop_idx = [active.pop(ii) for ii in idx]

            if Gram is None:
                # propagate dropped variable
                for ii in idx:
                    for i in range(ii, n_active):
                        X.T[i], X.T[i + 1] = swap(X.T[i], X.T[i + 1])
                        # yeah this is stupid
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]

                # TODO: this could be updated
                residual = y - np.dot(X[:, :n_active], coef[active])
                temp = np.dot(X.T[n_active], residual)

                Cov = np.r_[temp, Cov]
            else:
                for ii in idx:
                    for i in range(ii, n_active):
                        indices[i], indices[i + 1] = indices[i + 1], indices[i]
                        Gram[i], Gram[i + 1] = swap(Gram[i], Gram[i + 1])
                        Gram[:, i], Gram[:, i + 1] = swap(Gram[:, i],
                                                          Gram[:, i + 1])

                # Cov_n = Cov_j + x_j * X + increment(betas) TODO:
                # will this still work with multiple drops ?

                # recompute covariance. Probably could be done better
                # wrong as Xy is not swapped with the rest of variables

                # TODO: this could be updated
                residual = y - np.dot(X, coef)
                temp = np.dot(X.T[drop_idx], residual)
                Cov = np.r_[temp, Cov]

            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.)  # just to maintain size
            if verbose > 1:
                print("%s\t\t%s\t\t%s\t\t%s\t\t%s" % (n_iter, '', drop_idx,
                                                      n_active, abs(temp)))

    if return_path:
        # resize coefs in case of early stop
        alphas = alphas[:n_iter + 1]
        coefs = coefs[:n_iter + 1]

        if return_n_iter:
            return alphas, active, coefs.T, n_iter
        else:
            return alphas, active, coefs.T
    else:
        if return_n_iter:
            return alpha, active, coef, n_iter
        else:
            return alpha, active, coef

