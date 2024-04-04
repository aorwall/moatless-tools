# ... other code

def _separable(transform):
    """
    Calculate the separability of outputs.

    Parameters
    ----------
    transform : `astropy.modeling.Model`
        A transform (usually a compound model).

    Returns :
    is_separable : ndarray of dtype np.bool
        An array of shape (transform.n_outputs,) of boolean type
        Each element represents the separablity of the corresponding output.
    """
    def _combine_separability(left, right, operator):
        """
        Combine the separability information of two components using the specified operator.

        Parameters
        ----------
        left, right : ndarray
            The separability information of the left and right components.
        operator : function
            A function to combine the separability information.

        Returns
        -------
        combined : ndarray
            The combined separability information.
        """
        return operator(left, right)

    def _recursive_separable(model):
        """
        Recursively evaluate the separability of a model.

        Parameters
        ----------
        model : `astropy.modeling.Model`
            The model to evaluate.

        Returns
        -------
        is_separable : ndarray
            The separability information of the model.
        """
        if (model_matrix := model._calculate_separability_matrix()) is not NotImplemented:
            return model_matrix
        elif isinstance(model, CompoundModel):
            sepleft = _recursive_separable(model.left)
            sepright = _recursive_separable(model.right)
            return _combine_separability(sepleft, sepright, _operators[model.op])
        elif isinstance(model, Model):
            return _coord_matrix(model, 'left', model.n_outputs)

    return _recursive_separable(transform)