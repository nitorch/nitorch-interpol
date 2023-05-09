_convert_bound = {
    'zero': 'zero', 'zeros': 'zero',
    'replicate': 'replicate', 'repeat': 'replicate', 'nearest': 'replicate',
    'dct1': 'dct1', 'mirror': 'dct1',
    'dct2': 'dct2', 'reflect': 'dct2',
    'dst1': 'dst1', 'antimirror': 'dst1',
    'dst2': 'dst2', 'antireflect': 'dst2',
    'dft': 'dft', 'wrap': 'dft', 'circular': 'dft',
}


def convert_bound(bound):
    """Convert SciPy/PyTorch bound aliases to the FFT-based nitorch convention

    - zeros             -> zero
    - repeat, nearest   -> replicate
    - mirror            -> dct1
    - reflect           -> dct2
    - antimirror        -> dst1
    - antireflect       -> dst2
    - wrap, circular    -> dft
    """
    bound = bound.lower()
    if bound not in _convert_bound:
        raise ValueError('Unknown boundary condition: should be one of',
                         tuple(_convert_bound.keys()))
    return _convert_bound[bound]
