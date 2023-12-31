from enum import Enum
import re
import os
import pandas as pd
import numpy as np

class Parser(Enum):
    START = 0
    HEADER = 1
    DATA = 2


class Device(Enum):
    NO = 0
    SMALL_ACCEL = 1
    BIG_ACCEL = 2


def csv_process(filename):
    state = Parser.START
    device = Device.NO
    columns = []
    axis = None
    records = []

    df_small_accel = pd.DataFrame()
    df_big_accel = []


    with open(filename, 'r') as measurement:
        for line in measurement:
            if state == Parser.START:
                axis = None
                columns = []
                records = []

                if line.startswith('ADXL362'):
                    device = Device.SMALL_ACCEL
                    state = Parser.HEADER

                elif line.startswith('ADXL356'):
                    axis = re.findall(r'ADXL356\.(\w):', line)
                    device = Device.BIG_ACCEL
                    state = Parser.HEADER

            elif state == Parser.HEADER:
                columns = re.findall(r'(\w+)\[[^\]]+\]', line)
                if axis and len(columns) == 2:
                    columns[-1] += axis[0]
                print(columns)
                state = Parser.DATA


            elif state == Parser.DATA:
                if not line.strip():
                    state = Parser.START
                    df = pd.DataFrame(records, columns=columns)
                    if device == Device.SMALL_ACCEL:
                        df['t'] = df['t'].astype(float) - df['t'].astype(float).shift(1)
                        df['t'] = df['t'].fillna(0)
                        df_small_accel = pd.concat([df_small_accel, df])

                    elif device == Device.BIG_ACCEL:
                        df_big_accel.append(df)

                else:
                    records.append(re.findall(r'(-?\d+\.\d+)', line))

    X = []
    for i in range(0, len(df_big_accel), 3):
        x = df_big_accel[i]
        x = x.merge(df_big_accel[i+1], on='t')
        x = x.merge(df_big_accel[i+2], on='t')
        x['t'] = x['t'].astype(float) - x['t'].astype(float).shift(1)
        x['t'] = x['t'].fillna(0)
        X.append(x)

    df_big_accel = pd.concat(X)
    df_small_accel['t'] = df_small_accel['t'].cumsum()
    df_big_accel['t'] = df_big_accel['t'].cumsum()

    return df_small_accel, df_big_accel


if __name__ == '__main__':
    for filename in os.listdir('.'):
        if filename.endswith('.txt'):
            low_power, accel = csv_process(filename)
            low_power.to_csv('low_power_accel_' + filename, index=False)
            accel.to_csv('accel_' + filename, index=False)
