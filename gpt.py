from pathlib import Path
import tiktoken

from tiktoken.load import load_tiktoken_bpe

import torch

import json

from helpers import rms_norm
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
# Following tutorial here
# https://lightning.ai/fareedhassankhan12/studios/building-llama-3-from-scratch
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

MODEL_PATH = "/media/diskd/llm_models/llama/Meta-Llama-3-8B"
TOKENIZER_MODEL_PATH = f"{MODEL_PATH}/tokenizer.model"
PARAM_JSON_PATH =  f"{MODEL_PATH}/params.json"


print("\n")
print("\n")

print("VOCAB:\n")
print(" - Tensor: an n-dimensional array with n indices")
print(" - RoPE: Rotational Position Encoding")
print(" - SwigGLU: Swish-Gated Linear Unit")
print("\n")

print("Noted for Gly's own understanding\n")
print(" - In Llama3, positional encoding is being done after converting tokens to embeddings, embeddings to query vectors, query vectors to positional encoded embeddings again. The same for keys and values.")
print(" - Multiplying the query and key vectors/matrices will give us a score that maps each token to another. The score indicates the relationship betwen each token's query and key.")
print(" - In Llama models, they use 'Grouped Multi-Query Attention'. Which means in the implementation of the model itself, there are multiple heads with different query, key, value matrix transformations, which will yield different final value vectors for the input token. These final value vectors will then be merged together (not mathematically, they just become one big chonky vector) to represent the final final value vectors.")
print(" - RoPE: Dont fully understand this yet. I know what it does but not the math behind it.")
print(" - SwiGLU: No clues what the fuck is going on here like what the hell man.")
print("\n")





# Loading the tokenizer from Llama-3
TOKENIZER_MODEL = load_tiktoken_bpe(TOKENIZER_MODEL_PATH)
print(f"\r\nLength of tokenizer model:\r\n{len(TOKENIZER_MODEL)}")
print(f"\r\nType of tokenizer model:\r\n{type(TOKENIZER_MODEL)}")


print(f"\r\nFirst 10 items of tokenizer model:\r\n{dict(list(TOKENIZER_MODEL.items())[5600:5610])}")






# Loading a PyTorch model of Llama-3
LLM_MODEL = torch.load(f"{MODEL_PATH}/consolidated.00.pth", weights_only=True)

# Printing the first 11 layers of the model architecture
print(f"\r\nThe first 11 layers of the LLM Model:\r\n{list(LLM_MODEL.keys())[:11]}")






# Opening the parameters JSON file
with open(PARAM_JSON_PATH, "r") as f:
    CONFIG = json.load(f)


print(f"\r\nConfig:\r\n{CONFIG}")






# Storing model details into variable

# Dimension
DIM = CONFIG["dim"]
# Layers 
N_LAYERS = CONFIG["n_layers"]
# Heads 
N_HEADS = CONFIG["n_heads"]
# KV_heads 
N_KV_HEADS = CONFIG["n_kv_heads"]
# Vocabulary
VOCAB_SIZE = CONFIG["vocab_size"]
# Multiple
MULTIPLE_OF = CONFIG["multiple_of"]
# Multiplier
FFN_DIM_MULTIPLIER = CONFIG["ffn_dim_multiplier"]
# Epsilon
NORM_EPS = CONFIG["norm_eps"]
# RoPE
ROPE_THETA = torch.tensor(CONFIG["rope_theta"])







# Adding special tokens and token breakers
special_tokens = [
    "<|begin_of_text|>",  # Marks the beginning of a text sequence.
    "<|end_of_text|>",  # Marks the end of a text sequence.
    "<|reserved_special_token_0|>",  # Reserved for future use.
    "<|reserved_special_token_1|>",  # Reserved for future use.
    "<|reserved_special_token_2|>",  # Reserved for future use.
    "<|reserved_special_token_3|>",  # Reserved for future use.
    "<|start_header_id|>",  # Indicates the start of a header ID.
    "<|end_header_id|>",  # Indicates the end of a header ID.
    "<|reserved_special_token_4|>",  # Reserved for future use.
    "<|eot_id|>",  # Marks the end of a turn (in a conversational context).
] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]  # A large set of tokens reserved for future use.

# patterns based on which text will be break into tokens
tokenize_breaker = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"






# Initialize tokenizer with specified parameters
tokenizer = tiktoken.Encoding(

    # make sure to set path to tokenizer.model file
    name = TOKENIZER_MODEL_PATH,

    # Define tokenization pattern string
    pat_str = tokenize_breaker,

    # Assign BPE mergeable ranks from tokenizer_model of LLaMA-3
    mergeable_ranks = TOKENIZER_MODEL,

    # Set special tokens with indices
    special_tokens={token: len(TOKENIZER_MODEL) + i for i, token in enumerate(special_tokens)},
)






# Test converting prompt into numerical values
# Input prompt
prompt = "the answer to the ultimate question of life, the universe, and everything is "
print(f"\r\nUser prompt:\r\n{prompt}")

# Encode the prompt using the tokenizer and prepend a special token (128000)
tokens = [len(TOKENIZER_MODEL) + special_tokens.index("<|begin_of_text|>")] + tokenizer.encode(prompt)

print(f"\r\nPrompt encoded tokens:\r\n{tokens}")  # Print the encoded tokens

# Convert the list of tokens into a PyTorch tensor
tokens = torch.tensor(tokens)

# Decode each token back into its corresponding string
#prompt_split_as_tokens = [tokenizer.decode([int(t) for t in [token.item()]]) for token in tokens]
#
#print(f"\r\nPrompt decoded tokens:\r\n{prompt_split_as_tokens}")  # Print the decoded tokens

### OUTPUT
#[128000, 1820, 4320, 311, 279, 17139, 3488, 315, 2324, 11, 279, 15861, 11, 323, 4395, 374, 220]
#['<|begin_of_text|>', 'the', ' answer', ' to', ' the', ' ultimate', ' question', ' of', ' life', ',', ' the', ' universe', ',', ' and', ' everything', ' is', ' ']





# Creating embeddings from the initial word tokens (4096 per token to be exact)
# Define embedding layer with vocab size and embedding dimension
embedding_layer = torch.nn.Embedding(VOCAB_SIZE, DIM)

# Copy pre-trained token embeddings to the embedding layer
embedding_layer.weight.data.copy_(LLM_MODEL["tok_embeddings.weight"])

# Get token embeddings for given tokens, converting to torch.bfloat16 format
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)

# Print shape of resulting token embeddings
print(f"\r\nShape of unnormalized token embeddings:\r\n{token_embeddings_unnormalized.shape}")

##### OUTPUT ####
#torch.Size([17, 4096])
##### OUTPUT ####




# Normalizing the embeddings
# using RMS normalization and provided normalization weights
# We are using layers 0 because we are creating the first layer of the Llama-3 transformer architecture
token_embeddings = rms_norm(token_embeddings_unnormalized, 
                            LLM_MODEL["layers.0.attention_norm.weight"], NORM_EPS)


# Print the shape of the resulting token embeddings
print(f"\r\nShape of normalized token embeddings:\r\n{token_embeddings.shape}")

##### OUTPUT ####
#torch.Size([17, 4096])
##### OUTPUT ####




# Print the shapes of different weights
print(
    # Query weight shape
    f"\r\nLayer 0 query weights shape: {LLM_MODEL['layers.0.attention.wq.weight'].shape}",
    
    # Key weight shape
    f"\r\nLayer 0 key weights shape: {LLM_MODEL['layers.0.attention.wk.weight'].shape}",
    
    # Value weight shape
    f"\r\nLayer 0 value weights shape: {LLM_MODEL['layers.0.attention.wv.weight'].shape}",
    
    # Output weight shape
    f"\r\nLayer 0 output weights shape: {LLM_MODEL['layers.0.attention.wo.weight'].shape}"
)


##### OUTPUT ####
#torch.Size([4096, 4096]) # Query weight dimension
#torch.Size([1024, 4096]) # Key weight dimension
#torch.Size([1024, 4096]) # Value weight dimension
#torch.Size([4096, 4096]) # Output weight dimension
##### OUTPUT ####








# Retrieve query weight for the first layer of attention
q_layer0 = LLM_MODEL["layers.0.attention.wq.weight"]

# Calculate dimension per head
head_dim = q_layer0.shape[0] // N_HEADS

# Reshape query weight to separate heads
q_layer0 = q_layer0.view(N_HEADS, head_dim, DIM)

# Print the shape of the reshaped query weight tensor
print(f"\r\nReshaped weight queries to multi-head ({N_HEADS} heads):\r\n{q_layer0.shape}")


##### OUTPUT ####
#torch.Size([32, 128, 4096])
##### OUTPUT ####




# Extract the query weight for the first head of the first layer of attention
q_layer0_head0 = q_layer0[0]

# Print the shape of the extracted query weight tensor for the first head
print(f"Shape of the first head of the first layer:\r\n{q_layer0_head0.shape}")


##### OUTPUT ####
#torch.Size([128, 4096])
##### OUTPUT ####






# Finding the Query vector of each token by multiplying the query weights with the token embeddings
# Matrix multiplication: token embeddings with Transpose (T) of query weight for first head
q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T)

# Shape of resulting tensor: queries per token
print(f"\r\nShape of resulting tensor - Query vector (the tokens' according query vectors/embeddings. This happens before the positional encoding in Llama 3 and 2, instead of after in the transformer explained on starquest):\r\n{q_per_token.shape}")


##### OUTPUT ####
#torch.Size([17, 128])
##### OUTPUT ####







# We are spliting the query vectors list into pairs of query vectors list
# Reason is that we want to convert those pair into a complex (real and imaginary) number
# Then we perform rotational angle shift on those pairs
# Read more into RoPE and how it encodes positional data. As of know Gly does not understand the math behind it fully.

# Convert queries per token to float and split into pairs
q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)

# Print the shape of the resulting tensor after splitting into pairs
print(f"\r\nPair of query vectors per token shape:\r\n{q_per_token_split_into_pairs.shape}")

##### OUTPUT ####
#torch.Size([17, 64, 2])
##### OUTPUT ######## OUTPUT ####





# Generate values from 0 to 1 split into 64 parts
zero_to_one_split_into_parts = torch.tensor(range(len(q_per_token_split_into_pairs[0])))/len(q_per_token_split_into_pairs[0])

# Print the resulting tensor
print(f"\r\nZero to one split into {len(zero_to_one_split_into_parts)} parts:\r\n{zero_to_one_split_into_parts}")


##### OUTPUT ####
#tensor([0.0000, 0.0156, 0.0312, 0.0469, 0.0625, 0.0781, 0.0938, 0.1094, 0.1250,
#        0.1406, 0.1562, 0.1719, 0.1875, 0.2031, 0.2188, 0.2344, 0.2500, 0.2656,
#        0.2812, 0.2969, 0.3125, 0.3281, 0.3438, 0.3594, 0.3750, 0.3906, 0.4062,
#        0.4219, 0.4375, 0.4531, 0.4688, 0.4844, 0.5000, 0.5156, 0.5312, 0.5469,
#        0.5625, 0.5781, 0.5938, 0.6094, 0.6250, 0.6406, 0.6562, 0.6719, 0.6875,
#        0.7031, 0.7188, 0.7344, 0.7500, 0.7656, 0.7812, 0.7969, 0.8125, 0.8281,
#        0.8438, 0.8594, 0.8750, 0.8906, 0.9062, 0.9219, 0.9375, 0.9531, 0.9688,
#        0.9844])
##### OUTPUT ####

# Calculate frequencies using a power operation
freqs = 1.0 / (ROPE_THETA ** zero_to_one_split_into_parts)

# Display the resulting frequencies
print(f"\r\nFrequency to help encode positional information:\r\n{freqs}")


##### OUTPUT ####
#tensor([1.0000e+00, 8.1462e-01, 6.6360e-01, 5.4058e-01, 4.4037e-01, 3.5873e-01,
#        2.9223e-01, 2.3805e-01, 1.9392e-01, 1.5797e-01, 1.2869e-01, 1.0483e-01,
#        8.5397e-02, 6.9566e-02, 5.6670e-02, 4.6164e-02, 3.7606e-02, 3.0635e-02,
#        2.4955e-02, 2.0329e-02, 1.6560e-02, 1.3490e-02, 1.0990e-02, 8.9523e-03,
#        7.2927e-03, 5.9407e-03, 4.8394e-03, 3.9423e-03, 3.2114e-03, 2.6161e-03,
#        2.1311e-03, 1.7360e-03, 1.4142e-03, 1.1520e-03, 9.3847e-04, 7.6450e-04,
#        6.2277e-04, 5.0732e-04, 4.1327e-04, 3.3666e-04, 2.7425e-04, 2.2341e-04,
#        1.8199e-04, 1.4825e-04, 1.2077e-04, 9.8381e-05, 8.0143e-05, 6.5286e-05,
#        5.3183e-05, 4.3324e-05, 3.5292e-05, 2.8750e-05, 2.3420e-05, 1.9078e-05,
#        1.5542e-05, 1.2660e-05, 1.0313e-05, 8.4015e-06, 6.8440e-06, 5.5752e-06,
#        4.5417e-06, 3.6997e-06, 3.0139e-06, 2.4551e-06])
##### OUTPUT ####







# Convert queries per token to complex numbers
q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)

print(f"\r\nPair of query vectors as complex number per token shape:\r\n{q_per_token_as_complex_numbers.shape}")
# Output: torch.Size([17, 64])

# Calculate frequencies for each token using outer product of arange(17) and freqs
freqs_for_each_token = torch.outer(torch.arange(len(q_per_token_as_complex_numbers)), freqs)

print(f"\r\nFrequencies for each token:\r\n{freqs_for_each_token}")

# Calculate complex numbers from frequencies_for_each_token using polar coordinates
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

print(f"\r\nComplex numbers from frequencies for each token:\r\n{freqs_for_each_token}")

# Rotate complex numbers by frequencies
q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis

print(f"\r\nPair of query vectors as complex number per token rotated shape:\r\n{q_per_token_as_complex_numbers_rotated.shape}")
# Output: torch.Size([17, 64])

print(f"\r\nPair of query vectors as complex number per token rotated:\r\n{q_per_token_as_complex_numbers_rotated}")



# Convert rotated complex number back to pairs
q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers_rotated)

print(f"\r\nPair of query vectors as real number per token rotated shape:\r\n{q_per_token_split_into_pairs_rotated.shape}")
# Output: torch.Size([17, 64, 2])


# Convert rotated pairs of numbers back to normal list of real numbers
q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

print(f"\r\nQuery vectors as real number per token rotated shape:\r\n{q_per_token_split_into_pairs_rotated.shape}")
# Output: torch.Size([17, 128])




















