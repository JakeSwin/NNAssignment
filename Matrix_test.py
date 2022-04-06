import math

class matrix:
  def __init__(self, row_size, column_size):
    x = []
    y = []

    for i in range(row_size):
      x.append(0)
    for j in range(column_size):
      y.append(x.copy())

    self.matrix = y

  def __len__(self):
    return len(self.matrix)

  def __getitem__(self, key):
    return self.matrix[key]

  def print2dArray(self):
    for row in self.matrix:
      print([float(f"{x:.5f}") for x in row])

  def updateValue(self, x, y, val):
    self.matrix[x][y] = val

  def add(self, other):
    return [[self.matrix[i][j] + other[i][j] for j in range(len(self.matrix[0]))] for i in range(len(other))]

# def softmax(X):
#   return [math.e**i / sum([math.e**j for j in X]) for i in X]

# def sigmoid_2(X):
#   return [1 / (1 + math.e**(-x)) for x in X]

# def sigmoid(x):
#   return 1 / (1 + math.e**(-x))

# def applyActFunc(outputs):
#   print("Applied Sigmoid activation function")
#   return [sigmoid(o) for o in outputs]

# w1 = matrix(2, 3)
# w1.print2dArray()

# w1.updateValue(0,0,1.2)
# w1.updateValue(2,1,-0.01)
# w1.updateValue(1,0, 0.002)
# w1.print2dArray()

# print("-------")

# print(w1.add(w1))

print(sigmoid_2([0.215, 0.88, 0.167]))
print(applyActFunc([0.215, 0.88, 0.167]))
