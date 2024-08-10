import torch
from torch.nn import functional as F

from pavel_gpt.gpt2_config import GPT2Config
from pavel_gpt.gpt2_model import GPT2Model
from transformers import GPT2Tokenizer

model_path = "ai-forever/rugpt3small_based_on_gpt2"

config = GPT2Config(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

print(tokenizer.encode("Hello World!"))
exit()


model = GPT2Model.from_pretrained(model_path, config)
model.eval()
# model.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)


text = "Привет! Я большая языковая модель от Сбера, "
tokens = tokenizer.encode(text)
# tokens = [44, 42911, 16, 907, 11, 81, 407, 19893, 23206, 16]  # "Hello, I'm a language model,"
tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
tokens = tokens.unsqueeze(0).repeat(5, 1)  # (5, 8)
x = tokens
print(f"x: {x}")
# x.to('cuda')

# generate!
while x.size(1) < 30:  # max_length=30
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0]  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)



for i in range(5):
    tokens = x[i, :30].tolist()
    decoded = tokenizer.decode(tokens)
    print(">", decoded)

print(model)
exit()

# model.to(device)
