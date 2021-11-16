from pyteller.data import load_data
current_data, input_data = load_data('AL_Weather')
current_data.head()
## Find pyteller pipelines

from mlblocks.discovery import find_pipelines

find_pipelines('pyteller')
pipeline = 'pyteller.persistence.persistence'
## Instantiate the pyteller pipeline
from pyteller import Pyteller

pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=5,
    offset=0,
    time_column='valid',
    targets='tmpf',
    entity_column='station',
    entities='8A0',
)
context = pyteller.fit(current_data, output_=0)
train = context['X']
train.head()

# %% md



# %%

pyteller.fit(start_=1, **context)

# %% md

## Forecast

# %%

output = pyteller.forecast(data=input_data, postprocessing=False, predictions_only=False)
output['forecasts'].head()

# %% md

## Evaluate

# %%

scores = pyteller.evaluate(actuals=output['actuals'], forecasts=output['forecasts'],
                           metrics=['MAPE', 'sMAPE'])
scores.head()


pyteller.plot(output)



