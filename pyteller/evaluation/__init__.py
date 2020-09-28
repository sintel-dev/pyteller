from pyteller.evaluation.metrics import(
mean_absolute_error, mean_squared_error, accuracy_score
)



METRICS = {
    'MAPE': (mean_absolute_error),
    'MSE': (mean_squared_error),
    'accuracy': (accuracy_score)

}
