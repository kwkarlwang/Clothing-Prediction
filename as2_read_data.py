#%%
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

#%%
files = [
    os.path.join("./data/", file)
    for file in os.listdir("./data/")
    if file.endswith(".pkl")
]
files
#%%
data = []
#%%

for file in files:
    with open(file, "rb") as f:
        d = pickle.load(f)
        for model in d["models"]:
            curr_data = {}
            curr_data["model"] = model
            curr_model = d["models"][model]
            curr_data["desc"] = d["desc"]
            curr_data["avg recall"] = curr_model["macro avg"]["recall"]
            curr_data["BER"] = 1 - curr_data["avg recall"]
            curr_data["accuracy"] = curr_model["accuracy"]
            data.append(curr_data)


#%%
df = pd.DataFrame(data)
#%%
recall_table = pd.pivot_table(df, values="avg recall", index="desc", columns="model")


#%%
plt.figure(figsize=(15, 10))
sns.heatmap(recall_table, cmap="Blues", annot=True)
plt.title("Average recall of three classes")
#%%
ber_table = pd.pivot_table(df, values="BER", index="desc", columns="model")

#%%
plt.figure(figsize=(15, 10))
sns.heatmap(ber_table, cmap="Blues", annot=True)
plt.title("BER")

#%%

acc_table = pd.pivot_table(df, values="accuracy", index="desc", columns="model")

#%%
plt.figure(figsize=(15, 10))
sns.heatmap(acc_table, cmap="Blues", annot=True)
plt.title("accuracy")

#%%
recall_table