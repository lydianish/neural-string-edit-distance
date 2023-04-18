import torch, pdb
from transformers import BartTokenizer
from models import EditDistNeuralModelConcurrent

sources = ["helloooo word", "hw r ya"]
targets = ["hello world", "how are you"]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", add_space_prefix=True)
vocab = tokenizer.get_vocab()
device = torch.device('cuda')
model = EditDistNeuralModelConcurrent(vocab, vocab, device, model_type="bart")

source_batch = tokenizer(sources, return_tensors="pt", padding=True)
target_batch = tokenizer(targets, return_tensors="pt", padding=True)

class_loss = torch.nn.BCELoss()
kl_div_loss = torch.nn.KLDivLoss(reduction='none')
xent_loss = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters())


pdb.set_trace()
model(source_batch["input_ids"], target_batch["input_ids"])