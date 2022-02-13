class Perceptron(object):
    def __init__ (self, eta=0.01, N_iters=100000):
        self.lr = eta
        self.N_iters = N_iters
        self.weights = 0.0
        self.bias = 0.0

    def fit(self, x, y):
        for _ in range(self.N_iters):
            for i in range(len(x)):
                if self.predict(x[i]) != y[i]:
                    self.weights += self.lr * y[i] * x[i]
                    self.bias += self.lr * y[i]
    
    def net_input(self, x):
        return self.weights * x + self.bias

    def predict(self, x):
        return 1.0 if self.net_input(x) > 0.0 else 0.0

x = [1, 2, 3, 10, 20, -2, -10, -100, -5, -20]
y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

model = Perceptron()
model.fit(x, y)

test_x = [30, 40, -20, -60]
for i in range(len(test_x)):
    print("input {}: {}".format(test_x[i], model.predict(test_x[i])))

print(model.weights)
print(model.bias)
