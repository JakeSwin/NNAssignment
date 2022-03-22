import math

x = [
    [0.50, 1.00, 0.75],
    [1.00, 0.50, 0.75],
    [1.00, 1.00, 1.00],
    [-0.01, 0.50, 0.25],
    [0.50, -0.25, 0.13],
    [0.01, 0.02, 0.05]
]

y = [
  [1, 0],
  [1, 0],
  [1, 0],
  [0, 1],
  [0, 1],
  [0, 1]
]

w1 = [
    [0.74, 0.13, 0.68],  # x1
    [0.8, 0.4, 0.10],  # x2
    [0.35, 0.97, 0.96],  # x3
    [0.9, 0.45, 0.36]  # x0
]

w2 = [
    [0.35, 0.8],  # a4
    [0.50, 0.13],  # a5
    [0.90, 0.8],  # a6
    [0.98, 0.92]  # x0
]

lr = 0.1  # learning rate


def sigmoid(x):
  return 1 / (1 + math.e**(-x))


def applyActFunc(outputs):
  return [sigmoid(o) for o in outputs]

def getOutputs(input, weights):
  input += [1]
  print("Input: " + str(input))
  out = [0] * len(weights[0])

  for count, val in enumerate(input):
    weights_and_input = [val * w for w in weights[count]]
    print("Sum of %f and %s: %s"%(val, str(weights[count]), str(weights_and_input)))
    out = [x + y for x, y in zip(out, weights_and_input)]
  
  print("Output: " + str(out))
  return out

def forwardStep():
  layer1 = getOutputs(x[0], w1)
  layer1 = applyActFunc(layer1)
  layer2 = getOutputs(layer1, w2)

  error = [x - y for x, y in zip(y[0], layer2)]

  print("Error: " + str(error))

  out = [0] * 3
  for count in range(len(w2) - 1):
    out[count] = layer1[count] * (1 - layer1[count]) * ((error[0] * w2[count][0]) + (error[1] * w2[count][1]))

  print("Layer2 Weighted Error: " + str(out))

  inp = x[0] + [1]

  w1d = [[0] * 3] * 4
  for d1 in range(4):
    for d2 in range(3):
      w1d[d1][d2] = lr * out[d2] * inp[d1]

  w2d = [[0] * 2] * 4
  for d1 in range(4):
    for d2 in range(2):
      w2d[d1][d2] = lr * error[d2] * layer1[d1]

forwardStep()