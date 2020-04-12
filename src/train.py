"""

"""
import torch
import torch.nn as nn
from tqdm import tqdm
from BiLSTM import *
from data_loader import *
import config
import time
from sklearn import metrics

CUDA = torch.cuda.is_available()
np.random.seed(config.random_seed)
torch.manual_seed(config.random_seed)

if CUDA:
    gpu_cpu = torch.device('cuda')
    torch.cuda.manual_seed(config.random_seed)
else:
    gpu_cpu = torch.device('cpu')


def get_long_tensor(x):
    return torch.LongTensor(x).to(gpu_cpu)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(train: Examples, model: BiLSTM, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    model.train()
    for x, sm, sp, y in batch(train.shuffled(), config.batch_size):

        x = get_long_tensor(x)
        y = get_long_tensor(y)

        sm, sp = get_long_tensor(sm), get_long_tensor(sp)

        optimizer.zero_grad()

        predictions = model(x, sm, sp).squeeze(1)

        loss = criterion(predictions, y)

        acc = categorical_accuracy(predictions, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        count += 1
    return epoch_loss / count, epoch_acc / count


def evaluate(eval: Examples, model: BiLSTM, criterion):
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    model.eval()

    with torch.no_grad():
        for x, sm, sp, y in batch(eval.shuffled(), config.batch_size):
            x,y = get_long_tensor(x), get_long_tensor(y)
            sm, sp = get_long_tensor(sm), get_long_tensor(sp)


            predictions = model(x, sm, sp).squeeze(1)

            loss = criterion(predictions, y)

            acc = categorical_accuracy(predictions, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            count += 1
    return epoch_loss / count, epoch_acc / count


def evaluate_on_test(eval: Examples, model: BiLSTM, criterion, fi2l):
    epoch_loss = 0
    epoch_acc = 0
    count = 0
    model.eval()

    y_test = list()
    y_pred = list()

    with torch.no_grad():
        for x, sm, sp, y in batch(eval.shuffled(), config.batch_size):
            x, y = get_long_tensor(x), get_long_tensor(y)
            sm, sp = get_long_tensor(sm),  get_long_tensor(sp)

            predictions = model(x, sm, sp).squeeze(1)
            loss = criterion(predictions, y)

            y_test.extend(y.tolist())
            y_pred.extend(predictions.argmax(dim=1, keepdim=True).squeeze(1).tolist())

            acc = categorical_accuracy(predictions, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            count += 1

    for i, item in enumerate(y_test):
        y_test[i] = fi2l[item]
    for i, item in enumerate(y_pred):
        y_pred[i] = fi2l[item]

    print(metrics.classification_report(y_test, y_pred, zero_division=0))
    return epoch_loss / count, epoch_acc / count


def main(dl: Dataloader, model: BiLSTM):
    prev_best = float('inf')
    patience = 0
    decay = 0
    lr = config.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(config.max_epochs)):
        start_time = time.time()
        train_loss, train_acc = train(dl.train_examples, model, optimizer, criterion)
        dev_loss, dev_acc = evaluate(dl.dev_examples, model, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}')
        print(f'\t  Dev Loss: {dev_loss:.3f} |   Dev Acc: {dev_acc:.2f}')

        if dev_loss >= prev_best:
            patience += 1
            if patience == 3:
                lr *= 0.5
                #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                tqdm.write('Dev loss did not decrease in 3 epochs, halfing the learning rate')
                patience = 0
                decay += 1
        else:
            prev_best = dev_loss
            print('Save the best model')
            model.save()

        if decay >= 3:
            print('Evaluating model on test set')
            model.load()
            print('Load the best model')

            test_loss, test_acc = evaluate_on_test(dl.test_examples, model, criterion, dl.fi2l)

            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}')
            break


if __name__ == "__main__":
    print(f'mode {config.mode}')
    dl = Dataloader(config)
    model = BiLSTM(config, dl, gpu_cpu)
    main(dl, model)
