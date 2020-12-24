from pyteller.evaluation.metrics import (
    mean_absolute_error, mean_squared_error, MASE, sMAPE, MAPE, root_mean_square_error
)


METRICS_NORM = {
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
    'RMSE': root_mean_square_error,
    'MASE': MASE,
    'sMAPE': sMAPE,
    'MAPE': MAPE,

    # 'accuracy': accuracy_score

}
