import pandas as pd
import numpy as np
import scipy.io
import os
import re

filenames = [name.split('.')[0] for name in os.listdir('.') if name.endswith('.mat')]

for filename in sorted(filenames):
    print(filename)
    mat = scipy.io.loadmat(f'{filename}.mat')
    columns = [column for column in mat.keys() if not column.startswith('__')]

    records = {}
    longest = max([len(mat[col].T[0]) for col in columns])

    for col in columns:
        dt = mat[col].T[0]
        col_edit = re.sub(r'^X\d+_?', '', col)
        print(col_edit, len(dt))
        if len(dt) == 1:
            records[col_edit] = dt[0]
        elif len(dt) < longest:
            records[col_edit] = np.pad(dt, (0, longest - len(dt)))
        else:
            records[col_edit] = dt

    x = pd.DataFrame(records)
    x.to_csv(f"{filename}.csv", index=False)
