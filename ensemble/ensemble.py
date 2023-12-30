
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import accuracy_score, f1_score
np.set_printoptions(suppress=True, precision=3, edgeitems=10, linewidth=200)

test_x_df = pd.read_csv("/kaggle/input/latsis-experiments/test_without_labels.csv")["text"].tolist()
text_y_df = pd.read_csv("/kaggle/input/latsis-experiments/mock_test_labels.csv")["label"].tolist()

# test_x_df = test_x_df[:2000]
# text_y_df = text_y_df[:2000]

test_y = np.array(text_y_df, dtype=np.float32)
test_y = np.round(test_y)

def encode_texts(tokenizer, texts):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
    return torch.tensor(input_ids), torch.tensor(attention_masks)


stuff = os.listdir("/kaggle/working/ensemble/")
for folder in stuff:
    if folder.__contains__("."):# or not folder.__contains__("swissbert"):
        continue

    print("processing " + folder)

    model = torch.load("/kaggle/working/ensemble/" + folder + "/model.pt").cuda()
    tokenizer = torch.load("/kaggle/working/ensemble/" + folder + "/tokenizer.pt")
    if folder.__contains__("swissbert"):
        model.set_default_language("de_CH")

    model.eval()

    test_x, test_attention_mask = encode_texts(tokenizer, test_x_df)

    BS = 5
    predictions = []
    for i in range(0, len(test_x), BS):
        batch = test_x[i:i+BS].cuda()
        batch_attention_mask = test_attention_mask[i:i+BS].cuda()

        with torch.no_grad():  # Deactivate autograd engine to reduce memory usage
            prediction = model(batch, attention_mask=batch_attention_mask).logits.squeeze(-1)
        predictions.append(prediction.cpu().detach())  # Detach before moving to CPU

        del batch
        del batch_attention_mask
        del prediction
        torch.cuda.empty_cache()  # Free up memory
        if i % 100 == 0:
            print(i)
    
    predictions = torch.cat(predictions)
    predictions = predictions.detach().numpy()
    #print(predictions)

    #check f1
    predictions2 = np.round(F.sigmoid(torch.tensor(predictions)).numpy())
    # print(predictions2)
    # print(test_y)
    print("f1: " + str(f1_score(test_y, predictions2, average='macro')))

    del model
    del tokenizer
    del test_x
    del test_attention_mask
    torch.cuda.empty_cache()  # Free up memory
    gc.collect()  # Trigger Python garbage collection

    np.save("/kaggle/working/ensemble/" + folder + "/predictions.npy", predictions)




#average predictions
all_predictions = []
for folder in stuff:
    if folder.__contains__("."):
        continue
    if folder.__contains__("swissbert"):
        continue
    predictions = np.load("/kaggle/working/ensemble/" + folder + "/predictions.npy")
    all_predictions.append(predictions)

all_predictions = np.mean(all_predictions, axis=0)
all_predictions = F.sigmoid(torch.from_numpy(all_predictions)).numpy()
print(all_predictions.shape)

#save to a csv file
df = pd.DataFrame(all_predictions, columns=["label"])
df["text"] = test_x_df
df.to_csv("/kaggle/working/predictions.csv", index=False)

print("f1: " + str(f1_score(test_y, np.round(all_predictions), average='macro')))