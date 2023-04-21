import torch, pdb
from transformers import BartTokenizer
from models import EditDistNeuralModelConcurrent

sources = ["helloooo word", "hello world","hw r ya", "how are you"]
targets = ["hello world", "hello world","how are you", "how are you"]
labels = [0, 0, 1, 1]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", add_prefix_space=True)
vocab = tokenizer.get_vocab()
device = torch.device('cuda')
model = EditDistNeuralModelConcurrent(vocab, vocab, device, model_type="bart")

source_batch = tokenizer(sources, return_tensors="pt", padding=True)["input_ids"]#.to(device)
target_batch = tokenizer(targets, return_tensors="pt", padding=True)["input_ids"]#.to(device)

pdb.set_trace()
action_scores, expected_counts, logprobs, distorted_probs = model(source_batch, target_batch)
probs, probs_norm = model.probabilities(source_batch, target_batch)

