import math
import matplotlib.pyplot as plt

X = [
    [0.0, 1.0]
]

Y = [
    [1, 0]
]

# weights = [
#   [[0.5, 0.1],  # x0
#    [-0.2, 0.2],  # x1
#    [0.5, 0.3]],  # x2
#   [[0.7, 0.9],  # a4
#    [0.6, 0.8],  # a5
#    [0.2, 0.4]],  # x3
# ]

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

# dict = {
#   'epoch-1': {
#     'input-X-row': 0,
#     'results': {
#       'weights-before': [],
#       'weights-after': [],
#       'weight-change': [],
#       'error': []
#     }
#   }
# }

error_list = []

dict = { }

lr = 0.1 # learning rate

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

def meanSquaredError(raw_error):
  return 1/2 * sum([errors**2 for errors in raw_error])

def sigmoid(x):
  return 1 / (1 + math.e**(-x))

def applyActFunc(outputs):
  print("Applied Sigmoid activation function")
  return [sigmoid(o) for o in outputs]

def getOutputs(inputs, weights):
  inp = inputs + [1]
  printBlue(f"Input: {format1dArray(inp)}")
  out = [0] * len(weights[0])

  for count, val in enumerate(inp):
    weights_and_inp = [val * w for w in weights[count]]
    print(
        f"Sum of {val:.5f} and {format1dArray(weights[count])}: {format1dArray(weights_and_inp)}")
    out = [x + y for x, y in zip(out, weights_and_inp)]

  printBlue(f"Output: {format1dArray(out)}")
  print("------------------------------------")
  return out

def getWeightUpdates(inputs, weights, output):
  inp = inputs + [1]

  printBlue(f"Calculate weights with inputs: {format1dArray(inputs)}")

  wd = [[0 for i in range(len(weights[0]))].copy()
        for j in range(len(weights))]

  for d1 in range(len(wd)):
    for d2 in range(len(wd[0])):
      print(f"Calc: {lr} * {output[d2]:.5f} * {inp[d2]:.1f} = {(lr * output[d2] * inp[d1]):.5f}")
      wd[d1][d2] = lr * output[d2] * inp[d1]

  return wd

def oneEpoch(inputs, expected, weights1, weights2):
  layer1 = getOutputs(inputs, weights1)
  layer1 = applyActFunc(layer1)
  layer2 = getOutputs(layer1, weights2)

  error = [x - y for x, y in zip(expected, layer2)]

  print(f"Error: {format1dArray(error)}")

  out = [sum(o) for o in [[layer1[c1] * (1 - layer1[c1]) * dw for dw in [weight * error[c2] for c2, weight in enumerate(weights)]] for c1, weights in enumerate(weights2[:-1])]]

  print(f"Layer2 Weighted Error: {format1dArray(out)}")

  print("------------------------------------")
  w1d = getWeightUpdates(inputs, weights1, out)
  print("------------------------------------")
  w2d = getWeightUpdates(layer1, weights2, error)
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

  return (w1new, w2new, meanSquaredError(error))

for e in range(epochs):
  dict[f"epoch-{e}"] = {
    'weights-1': w1,
    'weights-2': w2,
  }
  for count, item in enumerate(X):
    w1, w2, error = oneEpoch(item, Y[count], w1, w2)

    error_list.append(error)

for e in dict:
  printBlue(f"------------------{e}------------------")
  print("weight1: ")
  print2dArray(dict[e]['weights-1'])
  print("weight2: ")
  print2dArray(dict[e]['weights-2'])

plt.plot(error_list)
plt.ylabel('Error')
plt.show()