import pandas as pd


def clean_serie(df: pd.DataFrame, key: str, spread=0.99):

    sales_list = df[key].tolist()
    sales_list.sort()  # we sort the sales list

    smallest_sale = max(sales_list[0:int((1-spread)/2*len(sales_list))])

    # the sales with the largest value that will also be ignored
    largest_sale = min(sales_list[
        int((1+spread)/2*len(sales_list)):
        len(sales_list)
    ])

    # eventually, we filter the df according to the data we just gathered
    df = df[(df[key] > smallest_sale) & (df[key] < largest_sale)]

    return df
