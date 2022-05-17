import numpy as np
from random_toolbox import pack_bits

print(pack_bits(b'\xa1\xfe\xff', bit_width=1))
print(pack_bits(b'\xa1\xfe\xff', bit_width=2))
print(pack_bits(b'\xa1\xfe\xff', bit_width=3))
print(pack_bits(b'\xa1\xfe\xff', bit_width=4))
print(pack_bits(b'\xa1\xfe\xff', bit_width=8))
