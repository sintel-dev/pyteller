from pyteller.benchmark import  benchmark
from pyteller.benchmark import VERIFIED_PIPELINES

pipelines = VERIFIED_PIPELINES
# pipelines= ['persistence']

datasets = {
    'taxi':'value',
    'AL_Weather':['tmpf','dwpf','relh','drct'],
}

results = benchmark(pipelines=pipelines)#, datasets=datasets)

