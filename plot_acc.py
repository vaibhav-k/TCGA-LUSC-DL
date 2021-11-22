import matplotlib.pyplot as plt

x = [1, 2, 3, 25, 30]
y = [
    41.152474880218506,
    52.0476231406886,
    58.4773952108712,
    58.8471239276891,
    58.85742574257425,
]
n = [
    "LSTM",
    "Vanilla CNN",
    "Transfer learning (VGG16) CNN",
    "MLP",
    "Logistic Regression",
]

fig, ax = plt.subplots()
plt.tick_params(axis="x", which="both", bottom=False,
                top=False, labelbottom=False)
ax.scatter(x, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
# plt.show()
plt.savefig("accuracies.png")
