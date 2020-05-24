import numpy as np

def _needlemann_wunsch(reference, transcript, gap_score=-100, sim_func=lambda x, y: 0 if x == y else -1):
    '''
    ' This function calculates the best alignments of two lists based on the
    ' NEEDLEMANN-WUNSCH algorithm
    ' Translated to python from http://www.biorecipes.com/DynProgBasic/code.html
    ' s,n := reference
    ' t,m := transcript
    ' if you want to reconstruct the results from the reference page use:
    '  def test_sim_func(a, b):
    '      sim_mat = [ 
    '      [2, -1, 1, -1],
    '      [-1, 2, -1, 1],
    '      [1, -1, 2, -1],
    '      [-1, 1, -1, 2]]
    '      sim_mat = np.array(sim_mat)
    '      id_a = "ACGT".find(a)
    '      id_b = "ACGT".find(b)
    '      return sim_mat[id_a,id_b]
    ' The default values for this function are derived from Dorotheas matlab adaption of this code
    ' for the GRID database. She used insertion and deletion scores instead of gap_score, but they 
    ' were set to equal values (-100) so I joined them again.
    '''
    n_ref = len(reference)
    n_trans = len(transcript)
    d_mat = np.zeros(shape=[n_trans + 1, n_ref + 1,])
    # Initialize the dynamic programming calculation using base conditions
    for idr in range(n_ref + 1):
        d_mat[0, idr] = gap_score * idr
    for idt in range(n_trans + 1):
        d_mat[idt, 0] = gap_score * idt

    # Calculate all D[i,j]
    for i in range(1, n_trans + 1):
        for j in range(1, n_ref + 1):
            match = d_mat[i-1,j-1] + sim_func(reference[j-1], transcript[i-1])
            gaps = d_mat[i,j-1] + gap_score
            gapt = d_mat[i-1,j] + gap_score
            d_mat[i,j] = np.max([match, gaps, gapt])

    # Do the Traceback to create the alignment
    i = n_trans
    j = n_ref
    ref_align = []
    trans_align = []
    while (i > 0) and (j > 0):
        if d_mat[i, j] - sim_func(reference[j - 1], transcript[i - 1]) == d_mat[i - 1, j - 1]:
            ref_align.insert(0, reference[j - 1])
            trans_align.insert(0, transcript[i - 1])
            i = i - 1
            j = j - 1
        elif d_mat[i, j] - gap_score == d_mat[i, j - 1]:
            ref_align.insert(0, reference[j - 1])
            trans_align.insert(0, None)
            j = j - 1
        elif d_mat[i, j] - gap_score == d_mat[i - 1, j]:
            ref_align.insert(0, None)
            trans_align.insert(0, transcript[i - 1])
            i = i - 1
        else:
            raise('should not happen')

    while j > 0:
        ref_align.insert(0, reference[j - 1])
        trans_align.insert(0, None)
        j = j - 1

    while i > 0:
        ref_align.insert(0, None)
        trans_align.insert(0, transcript[i - 1])
        i = i - 1
    
    return(ref_align, trans_align)

def needlemann_wunsch(reference, transcript):
    '''
    ' Actual evaluation of the aligned lists
    '''
    insertions = 0
    deletions = 0
    substitutions = 0
    ref_align, trans_align = _needlemann_wunsch(reference, transcript)
    for idr in range(len(ref_align)):
        if ref_align[idr] is None:
            insertions += 1
        elif trans_align[idr] is None:
            deletions += 1
        elif ref_align[idr] != trans_align[idr]:
            substitutions += 1
    accuracy = (len(reference) - deletions - insertions - substitutions) / len(reference)
    return (["nw.acc",
             "nw.len",
             "nw.del",
             "nw.ins",
             "nw.sub"], 
            [accuracy,
             len(reference),
             deletions,
             insertions,
             substitutions])
