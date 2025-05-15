import src.pikernel.dimension_1 as dim1
import src.pikernel.dimension_2 as dim2
import src.pikernel.utils as utils

def test_import():
    assert True

def test_sob_mat_1d():
    m = 1
    s = 1
    L = 1
    
    device = utils.device
    s_mat = dim1.Sob_matrix_1d(m, s, L, device)

    assert True
    