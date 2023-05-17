import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load and preprocess the data from the CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame.
    """
    data = pd.read_csv(file_path)
    data = data.drop(columns=['Series Name', 'Series Code', 'Country Code'])
    data = data.dropna()
    data = data.rename(columns={'Country Name': 'Country'})
    # data = data.drop([0, 1, 3])
    return data


def bar():
    """ Defining the function """
    M = data['Country']
    N = data['2010 [YR2010]']
    plt.figure(figsize=(10, 5))
    plt.bar(M, N, label='Military Expenditure', color = 'purple')
    plt.xlabel('Country', fontsize=14)
    plt.ylabel('% of GDP', fontsize=14)
    plt.title('Military expenditure (% of GDP) for 2010')
    plt.legend()
    plt.show()


def multiline():
    """ Defining the function """

    plt.figure(figsize=(10, 5))

    for country in data['Country']:
        x = list(range(2001, 2011))  # X-axis values (years)
        # Y-axis values (military expenditure)
        y = data.loc[data['Country'] == country,
                     '2001 [YR2001]':'2010 [YR2010]'].values[0]

        # Convert non-numeric values ('..') to NaN
        y = pd.to_numeric(y, errors='coerce')

        # Exclude NaN values from the plot
        mask = ~np.isnan(y)
        plt.plot(np.array(x)[mask], y[mask], label=country)

    plt.xlabel('Year', fontsize=14)
    plt.ylabel('% of GDP', fontsize=14)
    plt.title('Military expenditure (% of GDP)')
    plt.legend()
    plt.show()


def pie_chart():
    """ Defining the function """
    plt.figure(figsize=(8, 8))

    # Filter the data for the year 2002
    year = '2005 [YR2005]'
    data_2002 = data[['Country', year]].copy()

    # Convert non-numeric values ('..') to NaN
    data_2002[year] = pd.to_numeric(data_2002[year], errors='coerce')

    # Drop countries with NaN values
    data_2002.dropna(inplace=True)

    # Explode the highest percentage slice
    explode = [0] * len(data_2002)
    max_value_index = data_2002[year].idxmax()
    explode[max_value_index] = 0.1

    # Custom colors for the pie chart
    colors = ['gold', 'yellowgreen', 'lightcoral',
              'lightskyblue', 'orange', 'pink', 'purple']
    plt.pie(data_2002[year], labels=data_2002['Country'],
            autopct='%1.1f%%', explode=explode, colors=colors)
    plt.title(f'Military expenditure (% of GDP) - {year}')
    plt.axis('equal')
    plt.show()


# File path of the CSV file
file_path = "D:\python core\Military expenditure (% of GDP).csv"
data = load_data(file_path)
print(data)


bar()
multiline()
pie_chart()
