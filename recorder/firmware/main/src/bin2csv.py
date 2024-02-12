import struct

filename = input('Input binary file name: ')
csvfilename = input('Output tsv file name: ')
fmt = 'ffff'
struct_len = struct.calcsize(fmt)
struct_unpack = struct.Struct(fmt).unpack_from


with (open(filename, "rb") as fr, 
      open(csvfilename, "w") as fw):
    while True:
        data = fr.read(struct_len)
        if not data:
            break
        t, x, y, z = struct_unpack(data)
        print(f'{t:.0f}\t{x:.2f}\t{y:.2f}\t{z:.2f}', file=fw)

