from sklearn.metrics import mean_absolute_error

from pyteller.evaluation.metrics import (
    MAPE, MASE, mean_squared_error, root_mean_square_error, sMAPE)

METRICS_NORM = {
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
    'RMSE': root_mean_square_error,
    'MASE': MASE,
    'sMAPE': sMAPE,
    'MAPE': MAPE,

    # 'accuracy': accuracy_score

}
