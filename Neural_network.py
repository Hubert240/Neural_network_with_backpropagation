import numpy as np
import matplotlib.pyplot as plt

dane_uczenia = []
f = open("baza.txt", "r")
for x in f:
    x = x.strip()[1:-1]
    liczby = [int(num) for num in x.split(',')]
    dane_uczenia.append(liczby)
f.close()

dane_uczenia = np.array(dane_uczenia)
X_train = dane_uczenia[:, 1:]
y_train = dane_uczenia[:, 0]

dane_testowe = []

f = open("test.txt", "r")
for x in f:
    x = x.strip()[1:-1]
    liczby = [int(num) for num in x.split(',')]
    dane_testowe.append(liczby)
f.close()

dane_testowe = np.array(dane_testowe)
X_test = dane_testowe[:, 1:]
y_test = dane_testowe[:, 0]


class NeuralNetwork:
    def __init__(self, wielkosc_wejscia, k1, k2, wielkosc_wyjscia):
        self.W1 = np.random.randn(wielkosc_wejscia, k1)
        self.b1 = np.zeros((1, k1))
        self.W2 = np.random.randn(k1, k2)
        self.b2 = np.zeros((1, k2))
        self.W3 = np.random.randn(k2, wielkosc_wyjscia)
        self.b3 = np.zeros((1, wielkosc_wyjscia))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        c = np.max(self.z3, axis=1, keepdims=True)
        exp_scores = np.exp(self.z3 - c)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, lr):
        delta3 = self.probs
        delta3[range(X.shape[0]), y] -= 1
        dW3 = np.dot(self.a2.T, delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W3.T) * (self.a2 > 0)
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0)
        delta1 = np.dot(delta2, self.W2.T) * (self.a1 > 0)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

    def train(self, X, y, lr, num_epochs):
        for epoch in range(num_epochs):
            self.forward(X)
            self.backward(X, y, lr)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


wielkosc_wejscia = X_train.shape[1]
wielkosc_wyjscia = 10
liczba_epok = 7000
epochs = range(liczba_epok)
lr_vec = np.array([1e-3, 1e-4, 1e-5, 1e-6])
k1_vec = np.array([2,4,8,16])
k2_vec = k1_vec

dokladnosc_procentowa = 0
najlepsze_dokladnosc_lr =[]
najlepsze_k1=0
najlepsze_k2=0
najlepsza_dokladnosc = 0


k1_grid, k2_grid = np.meshgrid(k1_vec, k2_vec)

najlepsze_dokladnosc_k = []
najlepsze_dokladnosc_k_wykres= np.zeros((len(k1_vec), len(k2_vec)))

for k1_idx, k1 in enumerate(k1_vec):
    for k2_idx, k2 in enumerate(k2_vec):
        model = NeuralNetwork(wielkosc_wejscia, k1, k2, wielkosc_wyjscia)
        najlepsze_k = []

        for epoch in epochs:
            model.train(X_train, y_train, lr_vec[-1], 1)
            predictions = model.predict(X_test)
            dokladnosc = np.mean(predictions == y_test)
            dokladnosc_procentowa = dokladnosc * 100
            najlepsze_dokladnosc_k.append(dokladnosc_procentowa)

        najlepsze_dokladnosc_k_wykres[k2_idx, k1_idx] = dokladnosc_procentowa

        if najlepsze_dokladnosc_k[-1] > najlepsza_dokladnosc:
            najlepsza_dokladnosc = najlepsze_dokladnosc_k[-1]
            najlepsze_k1 = k1
            najlepsze_k2 = k2

        print("Dla k1:", k1)
        print("Dla k2:", k2)
        print("Dokładność %:", najlepsze_dokladnosc_k[-1])
        print()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(k1_grid.flatten(), k2_grid.flatten(), najlepsze_dokladnosc_k_wykres.flatten(), c='b', marker='o')
ax.set_xlabel('k1')
ax.set_ylabel('k2')
ax.set_zlabel('Dokładność (%)')
plt.show()

wykres_lr=[]
wykres_lr_vec=[]
for x in lr_vec:
    model = NeuralNetwork(wielkosc_wejscia, najlepsze_k1, najlepsze_k2, wielkosc_wyjscia)
    najlepsze_lr = []

    for epoch in epochs:
        model.train(X_train, y_train, x, 1)
        predictions = model.predict(X_test)
        dokladnosc = np.mean(predictions == y_test)
        dokladnosc_procentowa = dokladnosc * 100
        najlepsze_dokladnosc_lr.append(dokladnosc_procentowa)

    print("dla lr: ", x)
    print("dokładność %: ", najlepsze_dokladnosc_lr[-1])
    wykres_lr_vec.append(x)
    wykres_lr.append(najlepsze_dokladnosc_lr[-1])

plt.plot(wykres_lr_vec, wykres_lr)
plt.title("Zależność między lr a dokładnością")
plt.xlabel("lr")
plt.ylabel("Dokładność (%)")
plt.show()

model = NeuralNetwork(wielkosc_wejscia, najlepsze_k1, najlepsze_k2, wielkosc_wyjscia)
najlepsza_epoka=0
najlepsza_epo = 0
wszystkie_dokladnosci = []
dokladnosc_procentowa = 0

for epoch in epochs:
    model.train(X_train, y_train, lr_vec[2], 1)
    predictions = model.predict(X_test)
    dokladnosc = np.mean(predictions == y_test)
    dokladnosc_procentowa = dokladnosc * 100
    wszystkie_dokladnosci.append(dokladnosc_procentowa)
    if dokladnosc_procentowa > najlepsza_epoka:
        najlepsza_epoka =  dokladnosc_procentowa
        najlepsza_epo = epoch


print("Najwieksza dokladnosc: ", najlepsza_epoka)
print("epoka: ", najlepsza_epo)
plt.plot(epochs, wszystkie_dokladnosci)
plt.title("Zmiana dokladnosci dla epok")
plt.xlabel("Liczba epok")
plt.ylabel("Dokładność (%)")
plt.show()


