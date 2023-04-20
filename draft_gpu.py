import torch, pdb
from transformers import BartTokenizer
from models import EditDistNeuralModelConcurrent

sources = ["helloooo word", "hw r ya"]
targets = ["hello world", "how are you"]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", add_space_prefix=True)
vocab = tokenizer.get_vocab()
device = torch.device('cuda')
model = EditDistNeuralModelConcurrent(vocab, vocab, device, model_type="bart")

source_batch = tokenizer(sources, return_tensors="pt", padding=True)["input_ids"]#.to(device)
target_batch = tokenizer(targets, return_tensors="pt", padding=True)["input_ids"]#.to(device)

class_loss = torch.nn.BCELoss()
kl_div_loss = torch.nn.KLDivLoss(reduction='none')
xent_loss = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters())

action_scores, expected_counts, logprobs, distorted_probs = model(source_batch, target_batch)
score = model.probabilities(source_batch, target_batch)
