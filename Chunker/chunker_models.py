import pandas as pd

df = pd.read_csv("scraper/rolex_models.csv", sep=";")

grouped = df.groupby('Collection')


for name, group in grouped:
    count = len(group)
    chunked_file = 1

    counter = 0 
    data = []
    for i in range(count):
        
        if counter == 25:
            chunked_df = pd.DataFrame(data, columns=[f'{name} Descriptions'])
            chunked_df.to_csv(f'Chunker/Chunked/Descriptions/{name}_{chunked_file}.csv', index=False)
            counter = 0
            chunked_file += 1
            data = []
        
        data.append(f"The {group['Collection'].iloc[i]} collection (Reference Number: {group['Reference'].iloc[i]}) has a case size of {group['Size'].iloc[i]}, with a {group['Description'].iloc[i]} and {group['Complication'].iloc[i]} complications, all with a retail Price of {group['RRP'].iloc[i]}")
        counter += 1
    
    # If we break out of hte loop and there were less than 25 items, then we need to save that information
    if data:
        chunked_df = pd.DataFrame(data, columns=[f'{name} Descriptions'])
        chunked_df.to_csv(f'Chunker/Chunked/Descriptions/{name}_{chunked_file}.csv', index=False)

    
        