import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd


class DataIterator(object):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __call__(self, epochs=1):
        for n in range(1, epochs + 1):
            return self.iterate_epoch(self.X, self.y)

    def iterate_epoch(self, X, y):
        if self.batch_size is None:
            yield X, y
        shuffled = np.random.choice(len(X), len(X), replace=False)
        start = 0
        end = min(start + self.batch_size, len(shuffled))
        while start < len(shuffled):
            batch = shuffled[start:end]
            yield X[batch], y[batch]
            start += self.batch_size
            end = min(start + self.batch_size, len(shuffled))


def do_train(model, X_train, y_train, X_test, y_test, cost_function, epochs, batch_size, learn_rate, decay, exp_name):
    train_iterator = DataIterator(X_train, y_train, batch_size=batch_size)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs('dumps', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    train_stats = []
    previous_test_loss = np.inf
    early_stop_count = 0

    for e in range(epochs):
        with open(f'dumps/{exp_name}_{start_time}_epoch_{e}.dump', 'wb') as dump_file:
            pickle.dump(model, dump_file)

        epoch_loss = 0
        epoch_absolute_error = 0
        for b, (batch_X, batch_y) in enumerate(train_iterator()):
            batch_loss, batch_mae = do_batch(model, batch_X, batch_y, cost_function, learn_rate)
            epoch_absolute_error += len(batch_y) * batch_mae
            epoch_loss += np.array(batch_loss)
            loss_so_far = epoch_loss / (len(batch_y) + batch_size * (b))
            print(f'Epoch {e} | Batch {b} | MAE: {batch_mae:.3f} | learn rate: {learn_rate:.6f} | '
                  f'batch loss: {batch_loss/len(batch_y):.5f} | epoch_loss: {loss_so_far:.5f}')
            learn_rate *= decay

        # train_y_hat = cnn(X_train)
        # train_mae = np.mean(np.argmax(train_y_hat, axis=1) != y_train)
        # print(f'Epoch {e} train MAE:  {train_mae}')

        test_y_hat = model(X_test)
        test_mae = np.mean(np.argmax(test_y_hat, axis=1) != y_test)
        test_loss = cost_function(y_test, test_y_hat)
        print(f'Epoch {e} | Test MAE: {test_mae:.3f} | test loss: {test_loss/len(y_test):.5f}')

        train_stats.append({
            'epoch': e,
            'learn_rate_end': learn_rate,
            'train_loss': epoch_loss / len(X_train),
            'test_loss': test_loss / len(y_test),
            'train_mae': epoch_absolute_error / len(X_train),
            'test_mae': test_mae,
        })

        log = pd.DataFrame(train_stats).astype('float')
        log.to_csv(f'logs/{exp_name}_{start_time}.csv', index=False, float_format='%.5f')

        if test_loss / previous_test_loss >= 1.00:
            early_stop_count += 1
            if early_stop_count == 3:
                print ('Early stopping')
                break
        else:
            early_stop_count = 0
        previous_test_loss = np.array(test_loss)


def do_batch(model, batch_X, batch_y, cost_function, learn_rate):
        y_hat = model(batch_X)
        batch_mae = np.mean(np.argmax(y_hat, axis=1) != batch_y)
        loss = cost_function(batch_y, y_hat)
        loss.backward(learn_rate=learn_rate)
        return loss.detach(), batch_mae.detach()


