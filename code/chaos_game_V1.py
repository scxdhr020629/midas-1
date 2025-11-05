import numpy as np
from itertools import product  # Added for efficient k-mer generation

def generate_cgr_matrix(sequence, k=4, bases=["A", "C", "G", "U"], smoothing_alpha=1.0, log_scale=False):
    """
    Optimized Chaos Game Representation (CGR) matrix generation (fixed error).

    Fixes:
    - Use itertools.product for k-mer indexing (no tuple error).
    - Full Laplace smoothing: all k-mers get base prob >0.
    - Vectorized counting/mapping.

    Parameters:
    -----------
    sequence : str
        miRNA sequence.
    k : int, optional
        k-mer length (4 for 16x16).
    bases : list, optional
        Bases (RNA).
    smoothing_alpha : float, optional
        Laplace factor (1.0 default).
    log_scale : bool, optional
        log1p for CNN stability (False default).

    Returns:
    --------
    numpy.ndarray
        16x16 CGR matrix [0,1].
    """
    n = len(sequence)
    s = 2 ** k  # 16
    vocab_size = len(bases) ** k  # 256

    # k-mers list
    kmers = [sequence[i:i+k] for i in range(n - k + 1)]
    total_counts = len(kmers)

    # All k-mers in lex order
    kmer_list = list(product(bases, repeat=k))
    kmer_to_idx = {''.join(kmer): idx for idx, kmer in enumerate(kmer_list)}

    # Full Laplace: base prob for unobserved
    denom = total_counts + smoothing_alpha * vocab_size
    probs = np.full(vocab_size, smoothing_alpha / denom)

    # Update observed
    unique_kmers, counts = np.unique(kmers, return_counts=True)
    for kmer, count in zip(unique_kmers, counts):
        idx = kmer_to_idx.get(kmer)
        if idx is not None:
            probs[idx] = (count + smoothing_alpha) / denom

    # CGR: binary row/col from k-mer
    mat = np.zeros((s, s))
    for flat_idx in range(vocab_size):
        prob = probs[flat_idx]
        kmer = kmer_list[flat_idx]
        # Row: 0=A/C, 1=G/U
        row_bits = ''.join('0' if base in ['A', 'C'] else '1' for base in kmer)
        # Col: 0=A/U, 1=C/G
        col_bits = ''.join('0' if base in ['A', 'U'] else '1' for base in kmer)
        row_idx = int(row_bits, 2)
        col_idx = int(col_bits, 2)
        mat[row_idx, col_idx] += prob

    # Normalize (probs sum=1, mat too)
    if mat.sum() > 0:
        mat /= mat.sum()

    if log_scale:
        mat = np.log1p(mat)

    return mat
# Test example
# seq = "UUGCAUAGUCACAAAAGUGAUC"
# mat = generate_cgr_matrix(seq)
# print(mat)