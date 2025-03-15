from time import time
from tqdm import tqdm
from utils import *
import pandas as pd
from pot import pot_eval
from diagnosis import hit_att, ndcg
from pprint import pprint
import const

train_loader, test_loader, labels = load_dataset()
model, optimizer, scheduler, epoch, accuracy_list = load_model(dim)

## Prepare data
trainD, testD = next(iter(train_loader)), next(iter(test_loader))
trainO, testO = trainD, testD
trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

### Training phase
n = const.n; e = 0; start = time()
for e in tqdm(list(range(1, n+1))):
    lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
    accuracy_list.append((lossT, lr))
save_model(model, optimizer, scheduler, e, accuracy_list)

### Testing phase
torch.zero_grad = True
model.eval()
loss, _ = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

### Scores
preds = []
df = pd.DataFrame()
lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
for i in range(loss.shape[1]):
    lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
    result, pred = pot_eval(lt, l, ls); preds.append(pred)
    df = df.append(result, ignore_index=True)
lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
result.update(hit_att(loss, labels))
result.update(ndcg(loss, labels))
print(df)
pprint(result)
