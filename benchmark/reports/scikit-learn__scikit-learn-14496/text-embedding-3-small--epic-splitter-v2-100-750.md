**0 - /tmp/repos/scikit-learn/sklearn/cluster/optics_.py**:
```python
"""
    n_samples = X.shape[0]
    _validate_size(min_samples, n_samples, 'min_samples')
    if min_samples <= 1:
        min_samples = max(2, min_samples * n_samples)

    # Start all points as 'unprocessed' ##
    reachability_ = np.empty(n_samples)
    reachability_.fill(np.inf)
    predecessor_ = np.empty(n_samples, dtype=int)
    predecessor_.fill(-1)

    nbrs = NearestNeighbors(n_neighbors=min_samples,
                            algorithm=algorithm,
                            leaf_size=leaf_size,
                            metric=metric,
                            metric_params=metric_params,
                            p=p,
                            n_jobs=n_jobs)

    nbrs.fit(X)
    # Here we first do a kNN query for each point, this differs from
    # the original OPTICS that only used epsilon range queries.
    # TODO: handle working_memory somehow?
    core_distances_ = _compute_core_distances_(X=X, neighbors=nbrs,
                                               min_samples=min_samples,
                                               working_memory=None)
    # OPTICS puts an upper limit on these, use inf for undefined.
    core_distances_[core_distances_ > max_eps] = np.inf

    # Main OPTICS loop. Not parallelizable. The order that entries are
    # written to the 'ordering_' list is important!
    # Note that this implementation is O(n^2) theoretically, but
    # supposedly with very low constant factors.
    processed = np.zeros(X.shape[0], dtype=bool)
    ordering = np.zeros(X.shape[0], dtype=int)
    for ordering_idx in range(X.shape[0]):
        # Choose next based on smallest reachability distance
        # (And prefer smaller ids on ties, possibly np.inf!)
        index = np.where(processed == 0)[0]
        point = index[np.argmin(reachability_[index])]

        processed[point] = True
        ordering[ordering_idx] = point
        if core_distances_[point] != np.inf:
            _set_reach_dist(core_distances_=core_distances_,
                            reachability_=reachability_,
                            predecessor_=predecessor_,
                            point_index=point,
                            processed=processed, X=X, nbrs=nbrs,
                            metric=metric, metric_params=metric_params,
                            p=p, max_eps=max_eps)
    if np.all(np.isinf(reachability_)):
        warnings.warn("All reachability values are inf. Set a larger"
                      " max_eps or all data will be considered outliers.",
                      UserWarning)
    return ordering, core_distances_, reachability_, predecessor_
# ...
```
