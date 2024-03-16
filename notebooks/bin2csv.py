import struct
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Folder with .bin files')
parser.add_argument('res', type=int, help='Resolution in g') # default is 4g, sigma is 16g 
args = parser.parse_args()
print(f'Path: "{args.path}"')

fmt = 'iiii'
struct_len = struct.calcsize(fmt)
struct_unpack = struct.Struct(fmt).unpack_from


for root, subdirs, files in os.walk(args.path):
    for filename in files:
        if not filename.lower().endswith('.bin'):
            continue

        name, ext = os.path.splitext(filename)
        csvfilename = os.path.join(root, f'{name}.tsv')
        filename = os.path.join(root, filename)
        print(filename)

        if args.res == 4:
            resolution = 0.122  # 4g
        elif args.res == 16:
            resolution = 0.488

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
