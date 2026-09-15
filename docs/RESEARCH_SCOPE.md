# Public research scope

This release provides a compact implementation of the neural architecture used
to study structured interactions along a 1D reciprocal-space sequence.

For hidden representations \(x_i\) and \(x_j\), the relation module can form
attention-weighted low-order interactions of the form

\[
R_i = \sum_j A_{ij}\sum_{p,q} c_{pq}T_p(a_i)T_q(b_j),
\]

where \(A_{ij}\) is content- and relative-position-dependent attention and
\(T_p\) denotes a Chebyshev polynomial.  The implementation evaluates this
separable expression without materializing an unnecessary three-dimensional
pairwise feature tensor.

The public tests cover algebraic identities, masking, relative-position
behavior, parameter accounting, gradients, and shape preservation.  They are
software validation, not evidence of experimental or cross-material
reconstruction performance.

The physical reconstruction workflow, datasets, calibration procedure, and
experimental analysis remain internal.  No claim in this repository should be
read as releasing or independently validating those components.
