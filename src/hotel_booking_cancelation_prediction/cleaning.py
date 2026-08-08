import pandas as pd


def handling_missing_values(df) :
    '''
    imputing missing values 
    takes raw dataframe df 
    and returns back dataframe with zero missing values
    '''
    # copying dataframe
    df = df.copy()

    # working with children col
    if 'children' in df.columns :
        df['children'] = df['children'].fillna(0)

    # working with agent and country col 
    if 'agent' in df.columns :
        df['agent'] = df['agent'].astype('object').fillna('Unknown')
    if 'country' in df.columns :
        df['country'] = df['country'].fillna('Unknown')

    # working with company col 
    if 'company' in df.columns :
        df['has_company'] = df['company'].notnull().astype('int64')
        df = df.drop(columns = ['company'])

    return df


def removing_inconsistency(df): 
    '''
    taking each col of dataframe and 
    returning dataframe with removed extra spaces
    '''
    df = df.copy()
    # extracting categorical columns
    cat_col = df.select_dtypes(include = ['string' , 'object']).columns
    # removing extra whitespaces from dataframe
    for col in cat_col :
        df[col] = df[col].str.strip()
    # fixing inconsistence in meal columns 'undefined' to 'sc'
    if 'meal' in df.columns :
        df['meal'] = df['meal'].replace('Undefined' , 'SC')

    return df


def fix_datatypes(df) :
    '''
    taking df and converting datatypes of children and reservation_status_date
    children : float to int 
    reservation_status_data : str to datatime
    '''

    df = df.copy()
    # in children
    if 'children' in df.columns :
        df['children'] = df['children'].astype('int64')

    # in reservation_status_date
    if 'reservation_status_date' in df.columns :
        df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

    return df


def remove_duplicates(df) :
    '''
    removing duplicates and returning dataframe
    '''
    df = df.copy()
    return df.drop_duplicates()


def remove_unnecessary_col(df) :
    '''
    removed unnecessary col like reservation_status which direct tells whether the room was cancelled or not by
    flags like check-out or cancelled (values in this col)
    which will impact our target column 
    '''
    df = df.copy()
    if 'reservation_status' in df.columns:
        df = df.drop(columns = ['reservation_status'])

    return df


def handle_impossible_values(df) :
    '''
    Filters out zero-guest bookings and extreme ADR typos.
    '''
    df = df.copy()
    if {'adults', 'children', 'babies'}.issubset(df.columns):
        zero_guest_mask = (df['adults'] + df['children'] + df['babies']) == 0
        df = df[~zero_guest_mask]
        
    if 'adr' in df.columns:
        df = df[df['adr'] < 5000]
        
    return df


def clean_data(df):
    '''
    Executes the complete Data Cleaning pipeline sequentially.
    '''
    df = df.copy()
    df = handling_missing_values(df)
    df = removing_inconsistency(df)
    df = fix_datatypes(df)
    df = remove_duplicates(df)
    df = remove_unnecessary_col(df)
    df = handle_impossible_values(df)
    return df

def pandas_cleaning(df):
    '''
    Executes the complete Data Cleaning pipeline sequentially.
    '''
    df = df.copy()
    df = removing_inconsistency(df)
    df = remove_duplicates(df)
    df = remove_unnecessary_col(df)
    df = handle_impossible_values(df)
    return df


def prepare_for_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light pandas cleaning only — sklearn handles imputation/outliers later.
    Matches notebook 03 pipeline cell 2.
    """
    df = pandas_cleaning(df)

    if 'company' in df.columns:
        df['has_company'] = df['company'].notnull().astype('int64')
        df = df.drop(columns=['company'])

    return df