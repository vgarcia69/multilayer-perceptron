# multilayer-perceptron

A multilayer perceptron written from scratch with NumPy. It classifies breast tumors as malignant (`M`) or benign (`B`).

## Quick start

```bash
make split DATA=data/data.csv   # creates data/training.csv and data/validation.csv
make train                      # trains, saves data/trained_model.json and data/history_<optimizer>.json
make predict                    # loads the model and prints the validation loss
make compare                    # plots the training curves of every data/history_*.json
```

The first `make` call creates the `venv` and installs the dependencies (numpy, pandas, matplotlib, colorama).
Run `make clean` to delete the generated files, or `make fclean` to also delete the venv.

## Configure the model

Edit `srcs/train.py`:

```python
model = Model([
    DenseLayer(n_inputs, 24, activation='sigmoid', weights_initializer='he_uniform', optimizer='adam'),
    DenseLayer(24, 24,       activation='sigmoid', weights_initializer='he_uniform', optimizer='adam'),
    DenseLayer(24, 2,        activation='softmax', weights_initializer='he_uniform', optimizer='adam'),
])

model.train(X, Y,
            callbacks=[metrics, metrics_val, console_logger, history, graphics, early_stopping],
            loss="categorical_cross_entropy",
            learning_rate=0.001,
            batch_size=16,
            epochs=1000)
```

| Option                | Values                                             |
|-----------------------|----------------------------------------------------|
| `activation`          | `sigmoid`, `relu`, `softmax`                       |
| `weights_initializer` | `random`, `he_uniform`                             |
| `optimizer`           | `gradient_descent`, `adam`                         |
| `loss`                | `binary_cross_entropy`, `categorical_cross_entropy`|

## Callbacks

| Callback                                   | Role                                                                      |
|--------------------------------------------|---------------------------------------------------------------------------|
| `Metrics(X, Y, metrics_required=[...], prefix="")` | Computes `accuracy`, `loss`, `recall`, `precision` at each epoch  |
| `ConsoleLogger()`                          | Prints the metrics of each epoch                                          |
| `History()`                                | Records the metrics (`history.save(path)` writes them as JSON)            |
| `Graphics(history)`                        | Plots the accuracy and loss curves at the end of training                 |
| `EarlyStopping(patience=10)`               | Stops when `val_loss` stops improving and restores the best weights       |

Example with a validation set: `Metrics(X_val, Y_val, prefix="val_")` produces `val_accuracy` and `val_loss`.

> **Callback order matters:** put the `Metrics` callbacks **first** in the list. They fill the logs that `ConsoleLogger`, `History` and `EarlyStopping` read.
