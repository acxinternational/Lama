from datasets import load_dataset

datasetName1 = 'beyond_good_and_evil'
datasetPath1 = '../LocalDatasets/' + datasetName1

raw_datasets1 = load_dataset("Augustya07/neitzsche_beyond_good_and_evil_convo")
raw_datasets1.save_to_disk(datasetPath1)


'''
datasetPath2 = '../LocalDatasets/glue_mrpc'

raw_datasets2 = load_dataset("glue", "mrpc")
raw_datasets2.save_to_disk(datasetPath2)
'''