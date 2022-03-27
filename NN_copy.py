import math

X = [
    [0.0, 1.0]
]

Y = [
    [1, 0]
]

w1 = [
    [0.5, 0.1],  # x0
    [-0.2, 0.2],  # x1
    [0.5, 0.3]  # x2
]

w2 = [
    [0.7, 0.9],  # a4
    [0.6, 0.8],  # a5
    [0.2, 0.4],  # x3
]

lr = 0.1  # learning rate

epochs = 10

def printBlue(text): print("\033[96m{}\033[00m".format(text))
def printRed(text): print("\033[91m{}\033[00m".format(text))

def print2dArray(X):
  for row in X:
    print([float(f"{x:.5f}") for x in row])

def format1dArray(X):
  arr = []
  for i in X:
    arr += [float(f"{i:.5f}")]
  return arr


def sigmoid(x):
  return 1 / (1 + math.e**(-x))


def applyActFunc(outputs):
  print("Applied Sigmoid activation function")
  return [sigmoid(o) for o in outputs]


def getOutputs(input, weights):
  inp = input + [1]
  printBlue(f"Input: {format1dArray(inp)}")
  out = [0] * len(weights[0])

  for count, val in enumerate(inp):
    weights_and_inp = [val * w for w in weights[count]]
    print(f"Sum of {val:.5f} and {format1dArray(weights[count])}: {format1dArray(weights_and_inp)}")
    out = [x + y for x, y in zip(out, weights_and_inp)]

  printBlue(f"Output: {format1dArray(out)}")
  print("------------------------------------")
  return out


def oneEpoch(input, expected, weights1, weights2):
  layer1 = getOutputs(input, weights1)
  layer1 = applyActFunc(layer1)
  layer2 = getOutputs(layer1, weights2)

  error = [x - y for x, y in zip(expected, layer2)]

  print(f"Error: {format1dArray(error)}")

  out = [0] * 2
  for count in range(len(weights2) - 1):
    out[count] = layer1[count] * (1 - layer1[count]) * ((error[0] * weights2[count][0]) + (error[1] * weights2[count][1]))

  print(f"Layer2 Weighted Error: {format1dArray(out)}")
  print("------------------------------------")

  inp = input + [1]

  printBlue(f"Calculate weights w1 with inputs: {inp}")

  w1d = [
      [0, 0],
      [0, 0],
      [0, 0]
  ]

  for d1 in range(3):
    for d2 in range(2):
      print(f"Calc: {lr} * {out[d2]:.5f} * {inp[d2]:.1f} = {(lr * out[d2] * inp[d1]):.5f}")
      w1d[d1][d2] = lr * out[d2] * inp[d1]

  printBlue("Weight Updates 1:")
  print2dArray(w1d)

  print("------------------------------------")

  w2d = [
      [0, 0],
      [0, 0],
      [0, 0]
  ]

  inp = layer1 + [1]

  printBlue(f"Calculate weights w2 with inputs: {format1dArray(inp)}")

  for d1 in range(3):
    for d2 in range(2):
      print(f"Calc: {lr} * {error[d2]:.5f} * {inp[d1]:.5f} = {(lr * error[d2] * inp[d1]):.5f}")
      w2d[d1][d2] = lr * error[d2] * inp[d1]

  printBlue("Weight Updates 2:")
  print2dArray(w2d)

  print("------------------------------------")

  w1new = [[weights1[i][j] + w1d[i][j] for j in range(len(weights1[0]))] for i in range(len(weights1))]
  w2new = [[weights2[i][j] + w2d[i][j] for j in range(len(weights2[0]))] for i in range(len(weights2))]

  printBlue("New Weight 1s: ")
  print2dArray(w1new)

  printBlue("New Weight 2s: ")
  print2dArray(w2new)

  printRed("------------------------------------")
  printRed("------------------------------------")
  printRed("------------------------------------")

  return (w1new, w2new)

w1, w2 = oneEpoch(X[0], Y[0], w1, w2)
w1, w2 = oneEpoch(X[0], Y[0], w1, w2)

for e in range(epochs):
  for count, item in enumerate(X):
    w1, w2 = oneEpoch(item, Y[count], w1, w2)
