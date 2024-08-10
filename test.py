from pavel_gpt.gpt2_tokenizer import GPT2Tokenizer

tokens = [37954, 5, 629, 6174, 5075, 2232, 7063, 360, 385, 23331, 16, 225]
tokens = [44, 42911, 7578, 5]

model_path = "ai-forever/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
# print(tokenizer.bacov[37954])
# exit()

decoded = tokenizer.decode(tokens)
print(decoded)
