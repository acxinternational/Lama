from datasets import load_dataset

datasetName = 'beyond_good_and_evil'
datasetPath = '../LocalDatasets/' + datasetName
raw_datasets = load_dataset("Augustya07/neitzsche_beyond_good_and_evil_convo")
raw_datasets.save_to_disk(datasetPath)