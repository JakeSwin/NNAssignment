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

dict = {}

weights = [w1, w2]

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

def sigmoid(X):
  return [1 / (1 + math.e**(-x)) for x in X]

def getOutputs(inputs, weights):
  inp = inputs + [1]
  printBlue(f"Input: {format1dArray(inp)}")
  out = [0] * len(weights[0])

  for count, val in enumerate(inp):
    weights_and_inp = [val * w for w in weights[count]]
    print(f"Sum of {val:.5f} and {format1dArray(weights[count])}: {format1dArray(weights_and_inp)}")
    out = [x + y for x, y in zip(out, weights_and_inp)]

  printBlue(f"Output: {format1dArray(out)}")
  print("------------------------------------")
  return out

def getWeightUpdates(inputs, weights, output):
  inp = inputs + [1]

  printBlue(f"Calculate weights with inputs: {format1dArray(inputs)}")

  wd = [[0 for i in range(len(weights[0]))].copy() for j in range(len(weights))]

  for d1 in range(len(wd)):
    for d2 in range(len(wd[0])):
      print(f"Calc: {lr} * {output[d2]:.5f} * {inp[d2]:.1f} = {(lr * output[d2] * inp[d1]):.5f}")
      wd[d1][d2] = lr * output[d2] * inp[d1]

  return wd

def meanSquaredError(raw_error):
  return 1/2 * sum([errors**2 for errors in raw_error])

def oneEpoch(inputs, expected, weights1, weights2):
  layer1 = getOutputs(inputs, weights1)
  layer1 = sigmoid(layer1)
  layer2 = getOutputs(layer1, weights2)

  error = [x - y for x, y in zip(expected, layer2)]

  print(f"Error: {format1dArray(error)}")

  # out = [0] * 2
  # for count in range(len(weights2) - 1):
  #   out[count] = layer1[count] * (1 - layer1[count]) * ((error[0] * weights2[count][0]) + (error[1] * weights2[count][1]))

  # for c1, weights in enumerate(weights2[:-1]):
  #   dw = 0
  #   for c2, weight in enumerate(weights):
  #     dw += (weight * error[c2])
  #   out[c1] = layer1[c1] * (1 - layer1[c1]) * dw

  out = [sum(o) for o in [[layer1[c1] * (1 - layer1[c1]) * dw for dw in [weight * error[c2] for c2, weight in enumerate(weights)]] for c1, weights in enumerate(weights2[:-1])]]

  print(f"Layer2 Weighted Error: {format1dArray(out)}")
  print("------------------------------------")

  # inp = inputs + [1]

  # printBlue(f"Calculate weights w1 with inputs: {inp}")

  # w1d = [[0 for i in range(len(weights1[0]))].copy() for j in range(len(weights1))]

  # for d1 in range(len(w1d)):
  #   for d2 in range(len(w1d[0])):
  #     print(f"Calc: {lr} * {out[d2]:.5f} * {inp[d2]:.1f} = {(lr * out[d2] * inp[d1]):.5f}")
  #     w1d[d1][d2] = lr * out[d2] * inp[d1]

  # printBlue("Weight Updates 1:")
  # print2dArray(w1d)

  w1d = getWeightUpdates(inputs, weights1, out)

  print("------------------------------------")

  # w2d = [[0 for i in range(len(weights2[0]))].copy() for j in range(len(weights2))]

  # inp = layer1 + [1]

  # printBlue(f"Calculate weights w2 with inputs: {format1dArray(inp)}")

  # for d1 in range(len(w2d)):
  #   for d2 in range(len(w2d[0])):
  #     print(f"Calc: {lr} * {error[d2]:.5f} * {inp[d1]:.5f} = {(lr * error[d2] * inp[d1]):.5f}")
  #     w2d[d1][d2] = lr * error[d2] * inp[d1]

  # printBlue("Weight Updates 2:")
  # print2dArray(w2d)

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
  dict[f"epoch-{e}"] = {}
  for count, item in enumerate(X):
    w1, w2, error = oneEpoch(item, Y[count], w1, w2)

    dict[f"epoch-{e}"][f"input-row-{count}"] = {
        'error': error
    }

print(dict)

error_list = []

for e in dict:
  for inputs in dict[e]:
    error_list.append(dict[e][inputs]['error'])

print(error_list)

plt.plot(error_list)
plt.ylabel('Error')
plt.show()
