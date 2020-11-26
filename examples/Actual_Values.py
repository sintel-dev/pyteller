from pyteller.data import load_data

# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.data import load_data
current_data=load_data('../pyteller/data/Demand_train.csv')
import pandas as pd
Actual_max=pd.DataFrame()
cols=current_data.columns
for col in cols[1:]:
    signal = col

# Initialization settings: user doesn't change these once new data comes in
    from pyteller.core import Pyteller
    pyteller = Pyteller (
        pipeline='actual_averaged',
        # pipeline='actual_averaged',
        # pred_length=10,
        # offset=5,
        goal=None,
        goal_window=None
    )

    pyteller.fit(
        data=current_data,
        timestamp_col='DateTime',
        target_signals='T1_FWD',
        static_variables=None,
        # entity_col='station',
        train_size=.75)

    input_data=load_data('../pyteller/data/Demand_train.csv')
    forecast = pyteller.forecast(data=input_data)
    Actual_max[signal]=forecast.iloc[:,0]



Actual_max.to_csv('Actual_average_train.csv')
