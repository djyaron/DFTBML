import torch

import numpy as np

class SymEigB(torch.autograd.Function):
    """Solves standard eigenvalue problems for real symmetric matrices,
    and applies conditional or Lorentzian broadening to the eigenvalues
    during backpropagation to increase gradient stability.

    Notes
    -----
    Results from backward passes through eigen-decomposition operations
    tend to suffer from numerical stability [1]_  issues when operating
    on systems with degenerate eigenvalues. Fortunately,  the stability
    of such operations can be increased through the application of eigen
    value broadening. However, such methods will induce small errors in
    the returned gradients as they effectively mutate  the eigen-values
    in the backwards pass. Thus, it is important to be aware that while
    increasing the extent of  broadening will help to improve stability
    it will also increase the error in the gradients.

    Two different broadening methods have been  implemented within this
    class. Conditional broadening as described by Seeger [MS2019]_, and
    Lorentzian as detailed by Liao [LH2019]_. During the forward pass the
    `torch.symeig` function is used to calculate both the eigenvalues &
    the eigenvectors (U & :math: `\\lambda` respectively). The gradient
    is then calculated following:
    .. math::

    \bar{A} = U (\\bar{\Lambda} + sym(F \circ (U^t \\bar{U}))) U^T

    Where bar indicates a value's gradient passed in from  the previous
    layer, :math: `\\Lambda` is the diagonal matrix associated with the
    :math: `\bar{\\lambda}` values,  :math: `\\circ`  is the so  called
    Hadamard product, sym is the symmetrisation operator and F is:
    .. math::

        F_{i, j} = \frac{I_{i \ne j}{h(\\lambda_i - \\lambda_j}

    Where, for conditional broadening, h is:
    .. math::

        h(t) = max(|t|, \epsilon)sgn(t)

    and for, Lorentzian broadening:
    .. math::

        h(t) = \frac{t^2 + \epsilon}{t}

    The advantage of conditional broadening is that is is only applied
    when it is needed, thus the errors induced in the gradients will be
    restricted to systems whose gradients would be nan's otherwise.
    The Lorentzian method, on the other hand, will apply broadening to
    all systems, irrespective of whether or not it is necessary. Note
    that if the h function is a unity operator then this is identical
    to a standard backwards pass through an eigen-solver.


    .. [1] Where stability is defined as the propensity of a function to
           return nan values.

    References
    ----------
    .. [MS2019] Seeger, M., Hetzel, A., Dai, Z., & Meissner, E. Auto-
                Differentiating Linear Algebra. ArXiv:1710.08717 [Cs,
                Stat], Aug. 2019. arXiv.org,
                http://arxiv.org/abs/1710.08717.
    .. [LH2019] Liao, H.-J., Liu, J.-G., Wang, L., & Xiang, T. (2019).
                Differentiable Programming Tensor Networks. Physical
                Review X, 9(3).

    """

    KNOWN_METHODS = ['cond', 'lrnz']

    @staticmethod
    def forward(ctx, a, method='cb', bf=1E-12):
        """Finds the eigenvalues and eigenvectors of a real symmetric
        matrix using the torch.symeig function.

        Parameters
        ----------
        ctx : `torch.autograd.function.SymEigCB`
            This parameter is auto-parsed by PyTorch and is used to pass
            data from .forward() method to the .backward() method.
        a : `torch.tensor` [`float`]
            A real symmetric matrix whose eigenvalues & eigenvectors
            will be computed.
        method : `str`, optional
            Broadening method to used, availble options are:
                - "cond" for conditional broadening.
                - "lrnz" for Lorentzian broadening.
            See class doc-string for more info on these methods.
        bf : `float`, optional
            The degree of broadening (broadening factor).[Default=1E-12]

        Returns
        -------
        w : `torch.tensor` [`float`]
            The eigenvalues, in ascending order.
        v : `torch.tensor` [`float`]
            The eigenvectors.

        Warnings
        --------
        Under no circumstances should `bf` be a torch.tensor entity.
        The `method` and `bf` parameters MUST be passed as positional
        arguments and NOT keyword arguments.
        """
        # Check that the method is of a known type
        if method not in SymEigB.KNOWN_METHODS:
            raise ValueError('Unknown broadening method selected.')

        # Compute eigen-values & vectors using torch.symeig.
        w, v = torch.symeig(a, eigenvectors=True)

        # Save tensors that will be needed in the backward pass
        ctx.save_for_backward(w, v)

        # Save the broadening factor and the selecte broadening method.
        ctx.bf, ctx.bm = bf, method

        # Store dtype to prevent dtype mixing (don't mix dtypes)
        ctx.dtype = a.dtype

        # Return the eigenvalues and eigenvectors
        return (w, v)

    @staticmethod
    def backward(ctx, w_bar, v_bar):
        """Evaluates gradients of the matrix from which the eigenvalues
        and eigenvectors were taken.

        Parameters
        ----------
        w_bar : `torch.tensor` [`float`]
            Gradients associated with the the eigenvalues.
        v_bar : `torch.tensor` [`float`]
            Gradients associated with the eigenvectors.

        Returns
        -------
        a_bar : `torch.tensor` [`float`]
            Gradients associated with the `a` tensor.

        Notes
        -----
        See class doc-string for a more detailed description of this method.

        Todo
        ----
        - Find a better way to handle method selection, i.e. abstract
          broadening to separate class methods that are called based
          upon a KNOWN_METHODS lookup dictionary. [Priority: Low]
        - Create inplace operations to save memory. [Priority: Low]
        - Implement multi-system support. [Priority: Moderate]
        """
        """
        TODO (FH):
            Will need to take care of the batching dimensions, so there will be a lot of
            work to do that. First, use for loops to iterate over batches 
            (might increase back_propagation complexity)
            
            For the future:
                1) Verify correctness
                2) Increase backprop efficiency
        """
        # Equation to variable legend
        #   w <- \lambda
        #   v <- U

        # __Preamble__
        # Retrieve eigenvalues (w) and eigenvectors (v) from ctx
        # Make sure that the device is the same for both
        w, v = ctx.saved_tensors
        assert(w.device == v.device)
        
        #Pull out the device to use
        tensor_device = w.device
        
        if (len(w.shape) < 2) and (len(v.shape) < 3):
            # Unsqueeze all dimensions to have consistent for-loop approach
            w = w.unsqueeze(0)
            v = v.unsqueeze(0)
            w_bar = w_bar.unsqueeze(0)
            v_bar = v_bar.unsqueeze(0)
        
        #gradients list to hold tensors for concatenation
        gradients = list()
        batching_dim = v.shape[0]
        
        for i in range(batching_dim):
            # Retrieve, the broadening factor and convert to a tensor entity
            bf = torch.tensor(ctx.bf, dtype=ctx.dtype, device = tensor_device)
    
            # Retrieve the broadening method
            bm = ctx.bm
    
            # Form the eigenvalue gradients into diagonal matrix
            lambda_bar = w_bar[i].diag()
    
            # Identify the indices of the upper triangle of the F matrix
            tri_u = torch.triu_indices(*v[i].shape, 1) 
    
            # Construct the deltas
            deltas = w[i][tri_u[1]] - w[i][tri_u[0]] 
    
            # Apply broadening
            if ctx.bm == 'cond':  # <- Conditional broadening
                deltas = 1 / torch.where(torch.abs(deltas) > bf,
                                         deltas, bf) * torch.sign(deltas)
            elif ctx.bm == 'lrnz':  # <- Lorentzian broadening
                deltas = deltas / (deltas**2 + bf)
            else:  # <- Should be impossible to get here
                pass
    
            # Construct F matrix where F_ij = v_bar_j - v_bar_i; construction is
            # done in this manner to avoid 1/0 which can cause intermittent
            # hard-to-diagnose issues.
            F = torch.zeros(len(w[i]), len(w[i]), dtype=ctx.dtype, device = tensor_device) 
            F[tri_u[0], tri_u[1]] = deltas  # <- Upper triangle
            F[tri_u[1], tri_u[0]] -= F[tri_u[0], tri_u[1]]  # <- lower triangle
    
            # Construct the gradient following the equation in the doc-string.
            a_bar = v[i] @ (lambda_bar + sym(F * (v[i].T @ v_bar[i]))) @ v[i].T
            
            gradients.append(a_bar.unsqueeze(0))
        # Return the gradient. PyTorch expects a gradient for each parameter
        # (method, bf) hence two extra None's are returned
        if len(gradients) == 1:
            return gradients[0], None, None
        else:
            return torch.cat(gradients, 0), None, None

def eig_solve(a, b=None,  **kwargs):
    """Solves a standard or generalised eigenvalue problem of the form
    (Az = λBz) for a real symmetric matrix and will apply eigenvalue
    broadening during the backwards to improve gradient stability.

    Parameters
    ----------
    a : `torch.tensor` [`float`]
        Real symmetric matrix for which the eigenvalues & eigenvectors
        are to be computed. This is typically the Fock matrix.
    b : `torch.tensor` [`float`], optional
        Complementary positive definite real symmetric matrix for the
        generalised eigenvalue problem. Typically the overlap matrix.

    **kwargs
        Additional keyword arguments:
            ``scheme``:
                Selects the scheme to transform a generalised eigenvalue
                problem into a standard one. Available schemes are:
                    - "clsk" for Cholesky.
                    - "lodw" for Löwdin.
                (`str`) [DEFAULT='clsk']
            ``direct_inv``:
                If True then the matrix inversion will be computed directly
                rather than via a call to torch.solve. [DEFAULT=False]
            ``b_method``:
                Defines broadening method used, options are:
                    - "cond" for conditional broadening.
                    - "lrnz" for Lorentzian broadening.
                    - None for no broadening.
                See the SymEigB docstring for more information on the
                different methods (`str`, None). [DEFAULT='cb']
            ``bf``:
                Broadening factor to control broadening intensity. This
                is only relevant when ``b_method``!=None. (`float`)
                [DEFAULT=1E-12]

    Returns
    -------
    eval : `torch.tensor` [`float`]
        Eigenvalues, in ascending order.
    evec : `torch.tensor` [`float`]
        Corresponding orthonormal eigenvectors if the input ``a``.

    Notes
    -----
    A more detailed discussion of the impacts of broadening and the
    differences between the various broadening methods can be found
    in the docstring of the ``SymEigB`` class.

    Only generalised eigenvalue problems of the form Az = λBz can be
    solved by this function.

    Mathematical discussions regarding the Cholesky decomposition are
    made with reference to the  "Generalized Symmetric Definite
    Eigenproblems" chapter of Lapack. [Lapack]_


    References
    ----------
    .. [Lapack] www.netlib.org/lapack/lug/node54.html (Accessed 10/08/2020)


    Todo
    ----
    - Implement Löwdin Orthogonalisation scheme. [Priority: Low]
    - Pass through standard torch keyword arguments. [Priority: QOL]
    - Reduce functions overhead. [Priority: Low]

    """
    #a, b = None
    a = a.to_dense() if a.is_sparse else a
    b = b.to_dense() if b.is_sparse else b
    def call_solver(mat):
        """This deals with which eigen-solver should be called and what
        parameters should be passed though.

        Parameters
        ----------
        mat : `torch.tensor` [`float`]
            Matrix whose eigen-values/vectors are to be computed.

        Returns
        -------
        evals, evecs : `torch.tensor`
            The eigen-values/vectors
        """
        # Identify which broadening method is to be used
        b_method = kwargs.get('b_method', 'cond')

        if b_method is not None:  # If applying broadening
            return SymEigB.apply(mat, b_method, kwargs.get('bf', 1E-12))
        else:  # If not applying broadening
            return torch.symeig(mat, eigenvectors=True)

    # If this is a standard eigenvalue problem
    if b is None:
        # Then no special methods are required, just compute & return the
        # eigen-values/vectors
        return call_solver(a)

    # Otherwise either a Cholesky decomposition or Löwdin Orthogonalisation must
    # be performed. Identify which scheme is to be used:
    scheme = kwargs.get('scheme', 'clsk')

    if scheme == 'clsk':  # For Cholesky decomposition scheme
        # Perform Cholesky factorization (A = LL^{T}) of B to attain L
        l = torch.cholesky(b)

        # Compute the inverse of L:
        if kwargs.get('direct_inv', False):
            # Via the direct method if specifically requested
            l_inv = torch.inverse(l)
        else:
            # Otherwise compute via an indirect method (default)
            l_inv = torch.solve(torch.eye(a.shape[-1], dtype=a.dtype), l)[0]

        # To obtain C, perform the reduction operation C = L^{-1}AL^{-T}
        c = l_inv @ a @ l_inv.T

        # The eigenvalues of Az = λBz are the same as Cy = λy; hence:
        eval, evec_ = call_solver(c)

        # The eigenvectors, however, are not, so they must be recovered: z = L^{-T}y
        evec = torch.mm(l_inv.T, evec_)

        # Return the eigenvalues and eigenvectors
        return eval, evec

    elif scheme == 'lodw':  # For Löwdin Orthogonalisation scheme
        raise NotImplementedError('Löwdin is not yet implemented')

    else:  # If an unknown scheme was specified
        raise ValueError('Unknown scheme selected.')


def sym(x):
    """Symmetries the specified tensor.

    Parameters
    ----------
    x : `torch.tensor`
        The tensor to be symmetrised.

    Returns
    -------
    x_sym : `torch.tensor`
        The symmetrised tensor.

    Notes
    -----
    This dose not perform any checks for nans as pytorch tends to resolve
    this issue internally. Except when the tensor is all zeros.

    Todo
    ----
    - Add option to make the sym operation inplace. [Priority: Low]
    """

    return (x + x.T) / 2