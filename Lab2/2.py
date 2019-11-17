# read matrix
with open('fill.txt', 'r') as f:
    rows, cols = [int(x) for x in next(f).split()]
    p = [int(x) for x in next(f).split()]
    A = [[int(num) for num in line.split(' ')] for line in f]
print(A)
print(rows, cols, p)

# build complementary mat
Ac = [[0 for x in range(cols)] for y in range(rows)]
for i in range(rows):
    for j in range(cols):
        Ac[i][j] = 1 - A[i][j]

B = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]

X0 = set()
X0.add((p[0], p[1]))
X1 = set()
while True:
    # intersect Ac with Xk-1 + B
    for point in X0:
        for dir in B:
            candidate = (point[0] + dir[0], point[1] + dir[1])
            if Ac[candidate[0]][candidate[1]] == 1:
                X1.add(candidate)

    if X1 == X0:
        break

    X0 = X1 
    X1 = set()

# build solution as intersection
sol = [[0 for x in range(cols)] for y in range(rows)]
for i in range(rows):
    for j in range(cols):
        if (i, j) in X1 or A[i][j] == 1:
            sol[i][j] = 1

# write to file
with open('fill_out.txt', 'w') as outfile:
    for line in sol:
        for item in line:
            outfile.write("%s " % item)
        outfile.write("\n")