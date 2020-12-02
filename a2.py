#%% md
# Import necessary library package

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_json("./renttherunway_final_data.json.gz", lines=True, compression='gzip')


#%% [markdown]
# ## Data Exploration


#%% md
# ### Data Cleaning

#%% [markdown]
"""
There are 15 columns of data and it seems like there are 
substantial amount of missing data for weight, body type and 
bust size

"""
#%%
df.info()
#%%
for col in df.columns:
    print(f"missing {col} data: {df[col].isnull().sum()}")


#%% md
"""
By checking the first couple data, we realized that height, weight and bust size
should be transformed into numerical data
"""
#%%
df.head()

#%% md
"""
As of now, we can simply drop all the data that are incomplete

(MAYBE CHANGE LATER)
"""

#%%
df.dropna(inplace=True)

#%% md
# Transform weight to numerical value
#%%
# before
df["weight"].head()
#%%
# get rid of 'lbs'
df["weight"] = df["weight"].apply(lambda s: int(s[:-3]))
#%%
# after
df["weight"].head()

#%% [markdown]

# Transform height to inches
#%%
df["height"].head()
#%%
# Transform feet to inches
# total inches = (feet) * 12 + inches
df["height"] = df["height"].apply(lambda s: int(s[0]) * 12 + int(s[-3:-1]))

#%%
df["height"]


#%% md
# TODO: encode bust size, perhaps use use hot encoding for the lettering
#%%
df["bust size"].unique()


#%% md
# ### Data Visualizing

#%% md
"""
It seems like most of females are around 65 inches tall

"""
#%%
plt.figure(figsize=(10, 6))
ax = df["height"].plot.hist(bins=20, density=1)
ax.set_xlabel("Height (inches)")
print(f"mean height: {df['height'].mean()}")
print(f"median height: {df['height'].median()}")

#%% md
"""
The distribution of weight seems approximately normal

"""
#%%
plt.figure(figsize=(10, 6))
ax = df["weight"].plot.hist(bins=20, density=1)
ax.set_xlabel("Weight (lbs)")
print(f"mean weight: {df['weight'].mean()}")
print(f"median weight: {df['weight'].median()}")


#%% md
"""
Most are around 34 years old
"""
#%%

plt.figure(figsize=(10, 6))
ax = df["age"].plot.hist(bins=20, density=1)
ax.set_xlabel("Age")
print(f"mean age: {df['age'].mean()}")
print(f"median age: {df['age'].median()}")

#%%

#%% md
"""
TODO: pick a column to predict
"""

#%%
# seems like most of the ratings are very high
df["rating"].describe()
#%%
# most are fit
df["fit"].describe()

#%%
# size could be a good choice
df["size"].plot.hist(bins=20, density=1)
#%%
df["size"].unique()


#%%
