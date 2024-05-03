""" Convert .bin from SD card to .tsv for further processing
"""
import struct
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Folder with .bin files')
# Default resolution in firmware is 4g
parser.add_argument('res', type=int, help='Resolution in g') 
args = parser.parse_args()
print(f'Path: "{args.path}"')

fmt = 'iiii'
struct_len = struct.calcsize(fmt)
struct_unpack = struct.Struct(fmt).unpack_from
if args.res == 2:
    resolution = 0.061
elif args.res == 4:
    resolution = 0.122
elif args.res == 8:
    resolution = 0.244
elif args.res == 16:
    resolution = 0.488

for root, subdirs, files in os.walk(args.path):
    for filename in files:
        print(filename)
        if not filename.lower().endswith('.bin'):
            continue

        name, ext = os.path.splitext(filename)
        csvfilename = os.path.join(root, f'{name}.tsv')
        filename = os.path.join(root, filename)
        print(filename)

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
                print(f'{t}\t{x:.2f}\t{y:.2f}\t{z:.2f}', file=fw)
