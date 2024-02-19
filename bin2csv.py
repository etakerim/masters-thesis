import struct
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Folder with .bin files')
args = parser.parse_args()
print(f'Path: "{args.path}"')

fmt = 'iiii'
struct_len = struct.calcsize(fmt)
struct_unpack = struct.Struct(fmt).unpack_from

for filename in os.listdir(args.path):
    if not filename.lower().endswith('.bin'):
        continue
    print(filename)
    name, ext = os.path.splitext(filename)
    csvfilename = os.path.join(args.path, f'{name}.tsv')
    filename = os.path.join(args.path, filename)
    resolution = 0.061  # 2g

    with (open(filename, 'rb') as fr,
          open(csvfilename, 'w') as fw):
        while True:
            data = fr.read(struct_len)
            if not data:
                break
            t, x, y, z = struct_unpack(data)
            x *= resolution
            y *= resolution
            z *= resolution
            # if t != 0:
            print(f'{t}\t{x:.2f}\t{y:.2f}\t{z:.2f}', file=fw)
