import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_response(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # just so we return it back to main
    return extract_data(soup)

def extract_data(soup):
    table_data = []

    # Finds all the figures before the tables
    figures = soup.find_all('figure')

    for fig in figures:
        
        # Finds the header before the figure, which is the category of the table
        table_header = fig.find_previous(['h2', 'h3']).get_text(strip=True)

        table = fig.find('table')

        # Skip if no table after figure
        if not table:
            continue

        rows = table.find_all('tr')

        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) >= 2:
                row_data = [col.get_text(strip=True) for col in cols]
                full_row = [table_header] + row_data

                table_data.append(full_row)
            else:
                print('row skipped due to insufficient cells')
    return table_data


def main():
    url = 'https://www.luxurybazaar.com/grey-market/rolex-waitlist/'
    scraped_data = get_response(url)

    if scraped_data:
        df = pd.DataFrame(scraped_data, columns=['Category', 'Model', 'Average Wait Time', "Market Price VS Retail Price"])
        
        # Spliting the Average Wait Time into Min and Max Wait Times
        new_cols = df["Average Wait Time"].str.split(r' to | and | - |-|–', expand=True,n=1)

        df["Min Wait Time"] = new_cols[0]
        df["Max Wait Time"] = new_cols[1]
        
        df["Min Wait Time"] = df["Min Wait Time"].str.strip()
        df["Max Wait Time"] = df["Max Wait Time"].str.strip()

        df["Max Wait Time"] = df["Max Wait Time"].fillna(df["Min Wait Time"])

        df.drop(columns=["Average Wait Time"], inplace=True)


        # Sending over to CSV
        df.to_csv('rolex_waitlist.csv', index=False)
        df.info()
        return df
    
    else:
        print('No data scraped')
        return None

if __name__ == "__main__":
    main()