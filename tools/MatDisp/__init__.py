from IPython.core.interactiveshell import InteractiveShell
import numpy as np

IShell = InteractiveShell.instance()

def vec2str_lis(vector):
    return map(lambda x:str(round(x)),vector)

def mat2md(M:np.ndarray):
    if M.ndim > 2:
        return repr(M)
    head = r'\begin{bmatrix}'
    tail = r'\end{bmatrix}'
    if M.ndim == 1:
        content = r'\\'.join(vec2str_lis(M))
    if M.ndim == 2:
        content = r'\\'.join(['&'.join(vec2str_lis(vec)) for vec in M])
    return f'$${head}{content}{tail}$$'

IShell.display_formatter.formatters['text/markdown'].type_printers[np.ndarray]=mat2md
    
