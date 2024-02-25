#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import norm
from tabulate import tabulate
import time
import psutil

# Set up notebook to display multiple outputs in one cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# # Import Fund Data

# In[5]:


start_time = time.time()

data_active_1 = pd.read_csv('PCCOX.csv')
data_active_2 = pd.read_csv('PRILX.csv')
data_active_3 = pd.read_csv('RWMGX.csv')
data_passive = pd.read_csv('WFSPX.csv')
data_active_1.head()
data_active_1.info()
data_active_2.head()
data_active_2.info()
data_active_3.head()
data_active_3.info()
data_passive.head()
data_passive.info()

# Get execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Get memory usage
memory_info = psutil.Process().memory_info()
print(f"Memory Usage: {memory_info.rss / 1024 / 1024} MB")


# ## Fund Transformation

# In[6]:


start_time = time.time()

# compute the logarithmic returns of each of the funds
log_return_active_1 = np.log(1 + data_active_1['Adj Close'].pct_change())
log_return_active_2 = np.log(1 + data_active_2['Adj Close'].pct_change())
log_return_active_3 = np.log(1 + data_active_3['Adj Close'].pct_change())
log_return_passive = np.log(1 + data_passive['Adj Close'].pct_change())

# Get execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Get memory usage
memory_info = psutil.Process().memory_info()
print(f"Memory Usage: {memory_info.rss / 1024 / 1024} MB")


# In[7]:


start_time = time.time()

# plot the log returns

fig = plt.figure()
fig.subplots_adjust(hspace=0.6, wspace=0.6)

ax = fig.add_subplot(2, 2, 1)
sns.histplot(log_return_active_1.iloc[1:], ax=ax)
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.title('Log of PCCOX Returns')

ax = fig.add_subplot(2, 2, 2)
sns.histplot(log_return_active_2.iloc[1:], ax=ax)
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.title('Log of PRILX Returns')

ax = fig.add_subplot(2, 2, 3)
sns.histplot(log_return_active_3.iloc[1:], ax=ax)
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.title('Log of RWMGX Returns')

ax = fig.add_subplot(2, 2, 4)
sns.histplot(log_return_passive.iloc[1:], ax=ax)
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.title('Log of WFSPX Returns')

plt.show()

# Get execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Get memory usage
memory_info = psutil.Process().memory_info()
print(f"Memory Usage: {memory_info.rss / 1024 / 1024} MB")


# # Simulation

# ## Compute Drift & Variance

# In[8]:


start_time = time.time()

# compute the drift

mean_active_1 = log_return_active_1.mean()
var_active_1 = log_return_active_1.var()
drift_active_1 = mean_active_1 - (0.5*var_active_1)

mean_active_2 = log_return_active_2.mean()
var_active_2 = log_return_active_2.var()
drift_active_2 = mean_active_2 - (0.5*var_active_2)

mean_active_3 = log_return_active_3.mean()
var_active_3 = log_return_active_3.var()
drift_active_3 = mean_active_3 - (0.5*var_active_3)

mean_passive = log_return_passive.mean()
var_passive = log_return_passive.var()
drift_passive = mean_passive - (0.5*var_passive)

# Get execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Get memory usage
memory_info = psutil.Process().memory_info()
print(f"Memory Usage: {memory_info.rss / 1024 / 1024} MB")


# In[9]:


start_time = time.time()

# compute the variance and daily returns

days = 251
trials = 10000

stdev_active_1 = log_return_active_1.std()
Z_active_1 = norm.ppf(np.random.rand(days, trials)) 
daily_returns_active_1 = np.exp(drift_active_1 + stdev_active_1 * Z_active_1)

stdev_active_2 = log_return_active_2.std()
Z_active_2 = norm.ppf(np.random.rand(days, trials)) 
daily_returns_active_2 = np.exp(drift_active_2 + stdev_active_2 * Z_active_2)

stdev_active_3 = log_return_active_3.std()
Z_active_3 = norm.ppf(np.random.rand(days, trials)) 
daily_returns_active_3 = np.exp(drift_active_3 + stdev_active_3 * Z_active_3)

stdev_passive = log_return_passive.std()
Z_passive  = norm.ppf(np.random.rand(days, trials)) 
daily_returns_passive  = np.exp(drift_passive  + stdev_passive  * Z_passive)

# Get execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Get memory usage
memory_info = psutil.Process().memory_info()
print(f"Memory Usage: {memory_info.rss / 1024 / 1024} MB")


# In[10]:


daily_returns_active_1.shape


# ## Simulate Price Path for Trials

# In[11]:


start_time = time.time()

# calculate the stock price for every trial

price_paths_active_1 = np.zeros_like(daily_returns_active_1)
price_paths_active_1[0] = data_active_1.iloc[-1,5]
for t in range(1, days):
       price_paths_active_1[t] = price_paths_active_1[t-1]*daily_returns_active_1[t]
        
price_paths_active_2 = np.zeros_like(daily_returns_active_2)
price_paths_active_2[0] = data_active_2.iloc[-1,5]
for t in range(1, days):
       price_paths_active_2[t] = price_paths_active_2[t-1]*daily_returns_active_2[t]

price_paths_active_3 = np.zeros_like(daily_returns_active_3)
price_paths_active_3[0] = data_active_3.iloc[-1,5]
for t in range(1, days):
       price_paths_active_3[t] = price_paths_active_3[t-1]*daily_returns_active_3[t]
        
price_paths_passive = np.zeros_like(daily_returns_passive)
price_paths_passive[0] = data_passive.iloc[-1,5]
for t in range(1, days):
       price_paths_passive[t] = price_paths_passive[t-1]*daily_returns_passive [t]
        
# Get execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Get memory usage
memory_info = psutil.Process().memory_info()
print(f"Memory Usage: {memory_info.rss / 1024 / 1024} MB")


# In[12]:


#inspect price path array
price_paths_active_1
price_paths_active_1.shape


# ### Price Path Arrays to Dataframes

# In[13]:


start_time = time.time()

#array to dataframe
num_columns = 251

#price path 1
df1 = pd.DataFrame(price_paths_active_1)
df1 = df1.T
df1.columns = [f'Day {i}' for i in range(1, num_columns + 1)]

#price path 2
df2 = pd.DataFrame(price_paths_active_2)
df2 = df2.T
df2.columns = [f'Day {i}' for i in range(1, num_columns + 1)]

#price path 3
df3 = pd.DataFrame(price_paths_active_3)
df3 = df3.T
df3.columns = [f'Day {i}' for i in range(1, num_columns + 1)]

#price path passive
df_passive = pd.DataFrame(price_paths_passive)
df_passive = df_passive.T
df_passive.columns = [f'Day {i}' for i in range(1, num_columns + 1)]

# Get execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Get memory usage
memory_info = psutil.Process().memory_info()
print(f"Memory Usage: {memory_info.rss / 1024 / 1024} MB")

#inspect dataframes
df1.head()
df1.info()
df2.head()
df2.info()
df3.head()
df3.info()
df_passive.head()
df_passive.info()


# ## Calculate Return & Volatility

# In[14]:


start_time = time.time()

#PCCOX - data active 1
df1['Volatility'] = df1.std(axis = 1)
df1['Return'] = (df1['Day 251'] - df1['Day 1'])/df1['Day 1']

#PRILX - data active 2
df2['Volatility'] = df2.std(axis = 1)
df2['Return'] = (df2['Day 251'] - df2['Day 1'])/df2['Day 1']

#RWMGX - data active 3
df3['Volatility'] = df3.std(axis = 1)
df3['Return'] = (df3['Day 251'] - df3['Day 1'])/df3['Day 1']

#WFSPX - data passive
df_passive['Volatility'] = df_passive.std(axis = 1)
df_passive['Return'] = (df_passive['Day 251'] - df_passive['Day 1'])/df_passive['Day 1']

# Get execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Get memory usage
memory_info = psutil.Process().memory_info()
print(f"Memory Usage: {memory_info.rss / 1024 / 1024} MB")

#inspect changes
df1.head()
df2.head()
df3.head()
df_passive.head()


# # Calculate Average Return

# In[15]:


start_time = time.time()

PCCOX_returns = df1['Return'].mean()
PRILX_returns = df2['Return'].mean()
RWMGX_returns = df3['Return'].mean()
WFSPX_returns = df_passive['Return'].mean()

#create return table
table = [['Fund', 'Avg. Annual Return'],
        ['PCCOX', PCCOX_returns], 
        ['PRILX', PRILX_returns],
        ['RWMGX', RWMGX_returns],
        ['WFSPX', WFSPX_returns]]

print(tabulate(table, headers='firstrow', tablefmt='grid'))

# Get execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Get memory usage
memory_info = psutil.Process().memory_info()
print(f"Memory Usage: {memory_info.rss / 1024 / 1024} MB")


# In[16]:


#verify return values
PCCOX_returns 
PRILX_returns
RWMGX_returns
WFSPX_returns


# # Visualizations

# In[17]:


#plot return distributions

fig = plt.figure()
fig.subplots_adjust(hspace=0.8, wspace=0.8)

ax = fig.add_subplot(2, 2, 1)
sns.histplot(pd.DataFrame(price_paths_active_1).iloc[-1], ax=ax)
plt.xlabel("Price after 251 days")
plt.title('PCCOX Returns')

ax = fig.add_subplot(2, 2, 2)
sns.histplot(pd.DataFrame(price_paths_active_2).iloc[-1], ax=ax)
plt.xlabel("Price after 251 days")
plt.title('PRILX Returns')

ax = fig.add_subplot(2, 2, 3)
sns.histplot(pd.DataFrame(price_paths_active_3).iloc[-1], ax=ax)
plt.xlabel("Price after 251 days")
plt.title('RWMGX Returns')

ax = fig.add_subplot(2, 2, 4)
sns.histplot(pd.DataFrame(price_paths_passive).iloc[-1], ax=ax)
plt.xlabel("Price after 251 days")
plt.title('WFSPX Returns')

plt.show()


# ## PCCOX

# In[18]:


#plot 20 price paths
plt.figure(figsize=(15,6))
plt.plot(pd.DataFrame(price_paths_active_1).iloc[:,0:20])
plt.title("Simulated PCCOX Prices")
plt.xlabel("Days")
plt.ylabel("Price ($)")


# ## PRILX 

# In[19]:


#plot 20 price paths
plt.figure(figsize=(15,6))
plt.plot(pd.DataFrame(price_paths_active_2).iloc[:,0:20])
plt.title("Simulated PRILX Prices")
plt.xlabel("Days")
plt.ylabel("Price ($)")


# ## RWMGX

# In[20]:


#plot 20 price paths
plt.figure(figsize=(15,6))
plt.plot(pd.DataFrame(price_paths_active_3).iloc[:,0:20])
plt.title("Simulated RWMGX Prices")
plt.xlabel("Days")
plt.ylabel("Price ($)")


# ## WFSPX

# In[21]:


#plot 20 price paths
plt.figure(figsize=(15,6))
plt.plot(pd.DataFrame(price_paths_passive).iloc[:,0:20])
plt.title("Simulated WFSPX Prices")
plt.xlabel("Days")
plt.ylabel("Price ($)")


# ### Dataframes to CSV

# In[22]:


start_time = time.time()

df1.to_csv('PCCOX_returns.csv', index = False, header = True)
df2.to_csv('PRILX_returns.csv', index = False, header = True)
df3.to_csv('RWMGX_returns.csv', index = False, header = True)
df_passive.to_csv('WFSPX_returns.csv', index = False, header = True)

# Get execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# Get memory usage
memory_info = psutil.Process().memory_info()
print(f"Memory Usage: {memory_info.rss / 1024 / 1024} MB")

