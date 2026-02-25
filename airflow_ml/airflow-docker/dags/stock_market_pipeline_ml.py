"""
@author: Kunmi
"""
from kaggle.api.kaggle_api_extended import KaggleApi
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd 
import pyarrow as pa
from pyarrow import parquet as pq
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

os.environ.get['KAGGLE_USERNAME']
os.environ.get['KAGGLE_KEY']

def download_dataset(**kwargs):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('jacksoncrow/stock-market-dataset', path='/tmp', unzip=True)

    all_files = glob.glob('/tmp/*.csv')
    kwargs['ti'].xcom_push(key='raw_data_files', value=all_files)

def process_dataset(**kwargs):
    all_files = kwargs['ti'].xcom_pull(key='raw_data_files')
    data = pd.DataFrame(columns=['Symbol', 'Security Name', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    
    for file in all_files:
        symbol = os.path.splitext(os.path.basename(file))[0]
        try:
            df = pd.read_csv(file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            else:
                print(f"File {file} does not contain a 'Date' column. Skipping...")
                continue
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue
           
        
        df['Symbol'] = symbol
        df['Security Name'] = ''
        df = df[['Symbol', 'Security Name', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        data = data.append(df, ignore_index=True)
        
    data = data.astype({
        'Symbol': 'str',
        'Security Name': 'str',
        'Date': 'str',
        'Open': 'float',
        'High': 'float',
        'Low': 'float',
        'Close': 'float',
        'Adj Close': 'float',
        'Volume': 'float'
    })

        
    data.to_csv('/tmp/processed_data.csv', index=False)
    kwargs['ti'].xcom_push(key='processed_data_path', value='/tmp/processed_data.csv')
                   
def FE(**kwargs):
    processed_data_path = kwargs['ti'].xcom_pull(key='processed_data_path')
    processed_data = pd.read_csv(processed_data_path, parse_dates=['Date'])
    if processed_data['Date'].dtypes != 'datetime64[ns]':
        processed_data['Date'] = pd.to_datetime(processed_data['Date']) 
    
    processed_data.sort_values(['Symbol', 'Date'], inplace = True)
    processed_data['vol_moving_avg'] = processed_data.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(window=30).mean())
    processed_data['adj_close_rolling_med'] = processed_data.groupby('Symbol')['Adj Close'].transform(lambda x: x.rolling(window=30).median())

    processed_data.to_csv('/tmp/processed_data_fe.csv', index = False)
    kwargs['ti'].xcom_push(key='processed_data_fe_path', value = '/tmp/processed_data_fe.csv')


def convert_to_parquet(**kwargs):
    ti = kwargs['ti']
    processed_data_fe_path = ti.xcom_pull(key='processed_data_fe_path')
    processed_data_fe = pd.read_csv(processed_data_fe_path, parse_dates=['Date'])
    table = pa.Table.from_pandas(processed_data_fe)
    pq.write_table(table, '/home/airflow/processed_data_fe.parquet')


def read_parquet(**kwargs):
    df = pd.read_parquet('/home/airflow/processed_data_fe.parquet')
    print(df)

def preprocess_data(df):
    df = df.set_index('Date')
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled, scaler


def split_data(df):
    y=df['Close_Price_Pred']
    X=df.drop('Close_Price_Pred', axis=1)
    
    return X, y
'''
Rationale for LTSM Model: I used Youtube to consider 
alternatives to the Random Forest model,
and saw a comment about NEAT (Neuroevolution of Augmenting Topologies)
This neural architecture was  intriguing so I attempted 
to implement it, however via the interaction with ChatGPT
I discovered that due to it's lack of time series characteristics
which the LSTM model possesses, for this task, 
the objective for this problem would have a greater chance of operational
success and suitability via the LSTMM,
which I also discovered in abundance during the brief period of 
of Youtube ML predictive model research I conducted for this problem
'''
# Define LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(**kwargs):
    processed_data_fe_path = kwargs['ti'].xcom_pull(key='processed_data_fe_path')
    processed_data_fe = pd.read_csv(processed_data_fe_path, parse_dates=['Date'])
    processed_data_scaled, scaler = preprocess_data(processed_data_fe)
    joblib.dump(scaler, 'C:/Users/knet9/API-Services/scaler.pkl')

    X, y = split_data(processed_data_scaled)
    X = X.values.reshape((X.shape[0], 1, X.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_model((X_train.shape[1], X_train.shape[2]))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[es])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_test, y_test), callbacks=[es])
    model.save('/home/airflow/model.h5')
    kwargs['ti'].xcom_push(key='model_path', value='/home/airflow/model.h5')
    hist_df = pd.DataFrame(history.history)
    
    hist_csv_file = '/home/airflow/history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
            
default_args = {
    'owner':'airflow_ml',
    'depends_on_past': False, 
    'email_on_failure': False,
    'email_on_retry': False, 
    'retries': 3, 
    'retry_delay':timedelta(minutes=2),
    'start_date': datetime(2023, 4, 26)
}

dag = DAG(
    dag_id='stock_market_pipeline_ml',
    default_args=default_args, 
    description= 'Stock Market Pipeline ML',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

download_task = PythonOperator(
   task_id='download_dataset',
   python_callable=download_dataset,
   provide_context=True,
   dag=dag,
)

process_task = PythonOperator(
    task_id='process_dataset',
    python_callable=process_dataset,
    provide_context=True,
    dag=dag,
)

fe_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=FE,
    provide_context=True,
    dag=dag,
)

convert_task = PythonOperator(
    task_id='convert_to_parquet',
    python_callable=convert_to_parquet,
    provide_context=True,
    dag=dag,
)

read_task = PythonOperator(
    task_id='read_parquet',
    python_callable=read_parquet,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)


download_task >> process_task >> fe_task >> convert_task >> read_task >> train_model_task
