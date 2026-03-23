import pandas as pd

df = pd.read_csv('scraper/rolex_waitlist.csv')
grouped = df.groupby('Model')

for name, group in grouped:
    count = len(group)
    data = []
    # print(f'For the {group['Model'].iloc[0]}, the minimum wait time is {group['Min Wait Time'].iloc[0]} and the maximum wait time is {group['Max Wait Time'].iloc[0]}, the price change with retail in comparison to resell is {group['Market Price VS Retail Price'].iloc[0]}.')
    
    for i in range(count):
        data.append(f"For the {group['Model'].iloc[i]}, the minimum wait time is {group['Min Wait Time'].iloc[i]} and the maximum wait time is {group['Max Wait Time'].iloc[i]}, the price change with retail in comparison to resell is {group['Market Price VS Retail Price'].iloc[i]}.")
    

    chunked_df = pd.DataFrame(data, columns=[f'{name} Wait Time and Price Change Descriptions'])
    chunked_df.to_csv(f'Chunker/chunked/Waitlists/{name}.csv', index=False)

