import pandas as pd
import scipy.io

filename = "file"
mat = scipy.io.loadmat(f'{filename}.mat')
x = pd.DataFrame({
    'rpm': mat['X097_RPM'],
    'drive_end': mat['X097_DE_time'],
    'fan_end': mat['X097_FE_time']
})
x.to_csv(f"{filename}.csv", index=False)



