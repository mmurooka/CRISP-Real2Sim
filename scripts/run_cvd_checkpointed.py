#!/usr/bin/env python3
"""Execute the pinned CVD optimizer with a loss-equivalent memory-bounded kernel."""
import ast
from cvd_memory import SOURCE,consistency_loss

if __name__=='__main__':
    tree=ast.parse(SOURCE.read_text(),filename=str(SOURCE))
    matches=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='consistency_loss']
    if len(matches)!=1:
        raise RuntimeError('Upstream CVD layout changed; inspect adapter before running')
    tree.body=[n for n in tree.body if n is not matches[0]]
    print('Using checkpointed CVD pair loss; original image resolution and loss normalization retained.',flush=True)
    exec(compile(tree,str(SOURCE),'exec'),dict(__name__='__main__',__file__=str(SOURCE),
        consistency_loss=consistency_loss))
