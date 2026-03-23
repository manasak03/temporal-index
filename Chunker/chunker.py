import pandas as pd

def desc_chunker(file, sep):
    df = pd.read_csv(file, sep=sep)

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

def wait_chunker(file):
    df = pd.read_csv(file)
    grouped = df.groupby('Model')

    for name, group in grouped:
        count = len(group)
        data = []
        
        for i in range(count):
            data.append(f"For the {group['Model'].iloc[i]}, the minimum wait time is {group['Min Wait Time'].iloc[i]} and the maximum wait time is {group['Max Wait Time'].iloc[i]}, the price change with retail in comparison to resell is {group['Market Price VS Retail Price'].iloc[i]}.")
        

        chunked_df = pd.DataFrame(data, columns=[f'{name} Wait Time and Price Change Descriptions'])
        chunked_df.to_csv(f'Chunker/chunked/Waitlists/{name}.csv', index=False)

def main():
    desc_chunker(file='scraper/rolex_models.csv', sep=';')
    wait_chunker(file='scraper/rolex_waitlist.csv')

if __name__ == "__main__":
    main()