a = [[1,2,3], [4,5,6]]
b = [[1,2,3], [4,5,6]]
c = [[1,2,3], [5,4,6]]

x, y = a

for x1, y1 in zip(x, y):
    if x1 in b[0] and y1 in b[1]:
        print(True)