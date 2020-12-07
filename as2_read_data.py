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

for file in files:
    if "test" in file:
        continue
    with open(file, "rb") as f:
        d = pickle.load(f)
        for model in d["models"]:
            curr_data = {}
            curr_data["model"] = model
            curr_model = d["models"][model]
            desc = d["desc"]
            string = ""
            if "sym_diff" in desc:
                string += "sym_diff-"
            else:
                string += "union-"
            if "remove" in desc:
                string += "remove_stopwords-"
            else:
                string += "preserve_stopwords-"
            if "tfidf" in desc:
                string += "tfidf"
            else:
                string += "tf"

            curr_data["desc"] = string
            curr_data["avg recall"] = curr_model["macro avg"]["recall"]
            curr_data["BER"] = 1 - curr_data["avg recall"]
            curr_data["accuracy"] = curr_model["accuracy"]
            curr_data["fit recall"] = curr_model["0"]["recall"]
            curr_data["small recall"] = curr_model["1"]["recall"]
            curr_data["large recall"] = curr_model["2"]["recall"]
            data.append(curr_data)


#%%
df = pd.DataFrame(data)
#%%
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
fit_table = pd.pivot_table(df, values="fit recall", index="desc", columns="model")
plt.figure(figsize=(15, 8))
sns.heatmap(fit_table.transpose(), cmap="Blues", annot=True)
plt.title("fit recall")

#%%
small_table = pd.pivot_table(df, values="small recall", index="desc", columns="model")
plt.figure(figsize=(15, 8))
sns.heatmap(small_table.transpose(), cmap="Blues", annot=True)
plt.title("small recall")

#%%
large_table = pd.pivot_table(df, values="large recall", index="desc", columns="model")
plt.figure(figsize=(15, 8))
sns.heatmap(large_table.transpose(), cmap="Blues", annot=True)
plt.title("large recall")

#%%
reduce_ber_table = ber_table[
    [
        "naive bayes",
        "logistic regression, C=100, weights=None",
        "svm, kernel=linear, C=1",
        "decision tree, max_depth=5",
        "knn, n-neighbor=300",
    ]
]


#%%
reduce_ber_table.columns = [
    "naive bayes",
    "logistic regression",
    "svm",
    "decision tree",
    "knn",
]

#%%
reduce_ber_table.columns.name = "model"

#%%
reduce_ber_table.index.name = "text features"

#%%
sns.heatmap(reduce_ber_table, cmap="Blues_r", annot=True)
plt.title("Balanced Error Rate", fontsize=16)

#%%
with open("./data/test_result.pkl", "rb") as f:
    test = pickle.load(f)

#%%
test_data = []
for key in test:
    curr_model = test[key]
    curr_data = {}
    curr_data["model"] = key
    curr_data["avg recall"] = curr_model["macro avg"]["recall"]
    curr_data["BER"] = 1 - curr_data["avg recall"]
    curr_data["accuracy"] = curr_model["accuracy"]
    curr_data["fit recall"] = curr_model["0"]["recall"]
    curr_data["small recall"] = curr_model["1"]["recall"]
    curr_data["large recall"] = curr_model["2"]["recall"]
    test_data.append(curr_data)


#%%
df_test = pd.DataFrame(test_data)
df_test

#%%
test_table = pd.pivot_table(df_test, values="BER", columns="model")
test_table
#%%
test_table.columns = [
    "decision tree",
    "knn",
    "logistic regression",
    "naive bayes",
    "svm",
]

#%%
test_table = test_table.transpose()

#%%
test_table
#%%
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis("off")
ax.axis("tight")
ax.table(cellText=test_table.values, colLabels=df.columns, loc="center")
fig.tight_layout()
plt.show()
#%%
ber_table['naive bayes']