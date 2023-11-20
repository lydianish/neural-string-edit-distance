import torch, pdb
from transformers import BartTokenizer, AutoModel, AutoTokenizer
from models import EditDistNeuralModelConcurrent

sources = ["helloooo word", "hello world","hw r ya", "how are you"]
targets = ["hello world", "hello world","how are you", "how are you"]
labels = [0, 0, 1, 1]


device = torch.device('cuda')
pdb.set_trace()

model_path_1 = "/scratch/lnishimw/experiments/noise-normalization/neural-string-edit-distance/experiment_022/models/"
model_path_2 = "/scratch/lnishimw/experiments/noise-normalization/neural-string-edit-distance/experiment_022b/models/"

model_1 = AutoModel.from_pretrained(model_path_1)
model_2 = AutoModel.from_pretrained(model_path_2)
tokenizer = AutoTokenizer.from_pretrained(model_path_2)

source_batch = tokenizer(sources, return_tensors="pt", padding=True)["input_ids"]#.to(device)
target_batch = tokenizer(targets, return_tensors="pt", padding=True)["input_ids"]#.to(device)


