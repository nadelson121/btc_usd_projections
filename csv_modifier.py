import pandas as pd
df = pd.read_csv('~/BTC-USD.csv')
df_reorder = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']] # rearrange column here

with open('BTC-USD2.csv', 'r') as file:
    data = file.read()[:-1]
with open('BTC-USD2.csv', 'w') as file:
    file.write(data)

df_reorder.to_csv('~/BTC-USD2.csv', index=False)
