
import src.DataGenerator as dg

MAX_PWR = 5
LIST_LEN = 5
data = dg.DataGenerator(10, LIST_LEN, 2**MAX_PWR + 1)
data.generate_case()
# data.binarize_x()
rd = data.train_cases['x']
b_rd = data.train_cases['binary_x']

print(rd)
print(b_rd)
