# ff_demo.py
# Forward-Forward algorithm using Karpathy's micrograd Value engine
# Drop this into github.com/spicecoder/micrograd and run: python ff_demo.py

import random
import math
from micrograd.engine import Value

# -----------------------------------------------------------------------------
# 1. FF Layer: each layer is its own micrograd island
# -----------------------------------------------------------------------------
class LayerFF:
    def __init__(self, nin, nout, nonlin=True):
        self.W = [[Value(random.uniform(-1,1)) for _ in range(nin)] for _ in range(nout)]
        self.b = [Value(0.0) for _ in range(nout)]
        self.nonlin = nonlin

    def __call__(self, x):
        # x is a list of Values or floats
        out = []
        for row, bi in zip(self.W, self.b):
            act = sum((xi if isinstance(xi, Value) else Value(xi)) * wi for xi, wi in zip(x, row)) + bi
            out.append(act.relu() if self.nonlin else act)
        return out

    def parameters(self):
        return [p for row in self.W for p in row] + self.b

    def goodness(self, activations):
        """Hinton's goodness: sum of squared activations."""
        return sum(a * a for a in activations)

    def local_loss(self, pos_acts, neg_acts, theta=2.0):
        """
        We want:
          goodness(pos)  >> theta   (real data feels "energized")
          goodness(neg)  << theta   (fake data feels "flat")
        """
        g_pos = self.goodness(pos_acts)
        g_neg = self.goodness(neg_acts)

        # Soft hinge loss: push pos above theta, neg below theta
        # loss = log(1 + exp(theta - g_pos)) + log(1 + exp(g_neg - theta))
        # For micrograd simplicity, use: (theta - g_pos).relu() + (g_neg - theta).relu()
        loss = (theta - g_pos).relu() + (g_neg - theta).relu()
        return loss

    def learn(self, x_pos, x_neg, lr=0.05, theta=2.0):
        """One local forward-forward step. No global backprop."""
        # Forward both
        a_pos = self(x_pos)
        a_neg = self(x_neg)

        # Local objective
        loss = self.local_loss(a_pos, a_neg, theta)

        # Backward ONLY within this layer's graph
        loss.backward()

        # Update ONLY this layer's parameters
        for p in self.parameters():
            p.data -= lr * p.grad
            p.grad = 0.0   # micrograd manual zero

        return loss.data


# -----------------------------------------------------------------------------
# 2. Simple FF Network: stack of locally-greedy layers
# -----------------------------------------------------------------------------
class FFNet:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def learn(self, x_pos, x_neg, lr=0.05, theta=2.0):
        # Each layer sees its own positive/negative data
        # Negative data for layer i+1 comes from layer i's corrupted output
        current_pos = x_pos
        current_neg = x_neg

        total_loss = 0.0
        for layer in self.layers:
            loss = layer.learn(current_pos, current_neg, lr=lr, theta=theta)
            total_loss += loss

            # Generate next-layer inputs
            # Positive: clean forward pass
            current_pos = layer(current_pos)
            # Negative: corrupt the positive activations (add noise)
            current_neg = [a + Value(random.gauss(0, 0.5)) for a in current_pos]

        return total_loss


# -----------------------------------------------------------------------------
# 3. Demo: learn a simple 2D pattern (XOR-ish) without global labels
# -----------------------------------------------------------------------------
def demo():
    # Positive data: points near the corners of a square (two classes mixed)
    # The FF network must learn to "energize" on real structure
    positive_samples = [
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [0.9, 0.9],
        [-0.9, -0.9],
    ]

    # Negative data: random noise points
    def make_negative():
        return [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)]

    # Network: 2 -> 8 -> 4 -> 1
    net = FFNet([
        LayerFF(2, 8),
        LayerFF(8, 4),
        LayerFF(4, 1, nonlin=False),  # final layer linear for interpretability
    ])

    print("Training Forward-Forward network (no global backprop)...")
    for epoch in range(2000):
        # Pick one positive sample
        x_pos = positive_samples[epoch % len(positive_samples)]
        x_neg = make_negative()

        loss = net.learn(x_pos, x_neg, lr=0.03, theta=1.0)

        if epoch % 200 == 0:
            # Measure: average goodness on positive vs negative
            pos_out = net(x_pos)
            neg_out = net(x_neg)
            g_pos = sum(a*a for a in pos_out).data
            g_neg = sum(a*a for a in neg_out).data
            print(f"epoch {epoch:4d} | loss={loss:.4f} | g_pos={g_pos:.3f} | g_neg={g_neg:.3f}")

    print("\n--- Testing separation ---")
    test_pos = [1.0, 1.0]
    test_neg = make_negative()
    print(f"Positive sample {test_pos}: goodness = {sum(a*a for a in net(test_pos)).data:.3f}")
    print(f"Negative sample {test_neg}: goodness = {sum(a*a for a in net(test_neg)).data:.3f}")

    # A simple "classifier" emerges: threshold on final layer goodness
    print("\n--- Threshold classifier ---")
    for pt in positive_samples[:4]:
        g = sum(a*a for a in net(pt)).data
        label = "REAL" if g > 0.5 else "FAKE"
        print(f"  {pt} -> goodness={g:.3f} [{label}]")

    for _ in range(4):
        pt = make_negative()
        g = sum(a*a for a in net(pt)).data
        label = "REAL" if g > 0.5 else "FAKE"
        print(f"  {pt} -> goodness={g:.3f} [{label}]")


if __name__ == "__main__":
    demo()
