from pyteller.evaluation.metrics import (
    mean_absolute_error, mean_squared_error, MASE, sMAPE
)


METRICS_NORM = {
    'MAPE': mean_absolute_error,
    'MSE': mean_squared_error,
    'MASE': MASE,
    'sMAPE': sMAPE
    # 'accuracy': accuracy_score

}
