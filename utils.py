import torch
import torch.nn.functional as F


# === Configs ===

# toke ID for unused tokens in vocab
DEFAULT_DISTRIBUTION_TOKEN_ID = 151669

# 8 bins ranging [0, 32768), representing tokens remaining
# 8 bins per reward states. we have two (binary) so 8 * 2 = 16
LENGTH_BINS = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
NUM_LENGTH_BINS = len(LENGTH_BINS) - 1  

# binary reward vals
DEFAULT_REWARD_VALUES = [0.0, 1.0]



# total number of bins
def get_num_bins(num_reward_states: int = 2) -> int:
    return NUM_LENGTH_BINS * num_reward_states

# map tokens remaining to length bin index
def get_length_bin(tokens_remaining: int) -> int:
    for i in range(NUM_LENGTH_BINS):
        if tokens_remaining < LENGTH_BINS[i + 1]:
            return i
    return NUM_LENGTH_BINS - 1  # clamp to last bin


# Maps discrete reward values (like [0.0, 1.0]) to bin boundaries (e.g., [-inf, 0.5, inf])
# so continuous rewards are assigned to bins based on which interval they fall into
def compute_reward_bin_edges(reward_values: list[float]) -> list[float]:
    edges = [float("-inf")]
    for i in range(len(reward_values) - 1):
        edges.append((reward_values[i] + reward_values[i + 1]) / 2)
    edges.append(float("inf"))
    return edges



# map reward value to reward bin index
def get_reward_bin(reward: float, reward_values: list[float] = DEFAULT_REWARD_VALUES) -> int:
    
    edges = compute_reward_bin_edges(reward_values)
    
    # find bin where edges[i] <= reward < edges[i+1]
    for i in range(len(reward_values)):
        if reward < edges[i + 1]:
            return i
    return len(reward_values) - 1




# map (tokens_remaining, reward) join distribution to a single bin index for the joint distribution
# which is basically length bin index + reward bin (which there are less than of length bins) * the number of length bins
def get_joint_bin(tokens_remaining: int, reward: float, reward_values: list[float] = DEFAULT_REWARD_VALUES) -> int:

    length_bin = get_length_bin(tokens_remaining)
    reward_bin = get_reward_bin(reward, reward_values)
    return length_bin + reward_bin * NUM_LENGTH_BINS


# === Logit Extraction ===

# find logits that represent join distribution [default_distribution_token_id, default_distribution_token_id + num_bins)
def extract_joint_logits(logits: torch.Tensor, distribution_token_id: int = DEFAULT_DISTRIBUTION_TOKEN_ID, num_reward_states: int = 2,) -> torch.Tensor:
    num_bins = get_num_bins(num_reward_states)

    vocab_size = logits.shape[-1]

    # sanity check

    assert distribution_token_id + num_bins <= vocab_size, (
        f"Reserved tokens exceed vocab: {distribution_token_id} + {num_bins} > {vocab_size}"
    )

    return logits[..., distribution_token_id : distribution_token_id + num_bins]


# apply softmax to logits to
def get_joint_probs(logits: torch.Tensor, distribution_token_id: int = DEFAULT_DISTRIBUTION_TOKEN_ID, num_reward_states: int = 2,) -> torch.Tensor:
    
    joint_logits = extract_joint_logits(logits, distribution_token_id, num_reward_states)
    probs = F.softmax(joint_logits.float(), dim=-1)
    
    # take flat logits and reshape into rows = reward val and column = length bins
    *leading, num_bins = probs.shape
    return probs.view(*leading, num_reward_states, NUM_LENGTH_BINS)


# === Expected Value Computation ===


# [P(0), P(1)] * [0, 1] = [0, P(1)], then sum that
def get_expected_reward(joint_probs: torch.Tensor, reward_values: list[float] = DEFAULT_REWARD_VALUES, device: torch.device = None) -> torch.Tensor:

    if device is None:
        device = joint_probs.device
    
    reward_centers = torch.tensor(reward_values, dtype=torch.float32, device=device)
    
    # marginalize over length (so we only get P(reward)
    reward_marginal = joint_probs.sum(dim=-1) 
    
    return (reward_marginal * reward_centers).sum(dim=-1)


def get_expected_length(joint_probs: torch.Tensor, device: torch.device = None) -> torch.Tensor:

    if device is None:
        device = joint_probs.device
    
    # midpoint of each bin
    length_centers = torch.tensor(
        [(LENGTH_BINS[i] + LENGTH_BINS[i + 1]) / 2 for i in range(NUM_LENGTH_BINS)],
        dtype=torch.float32,
        device=device,
    )
    
    # marginalize over reward so we only get P(tokens reamining)
    length_marginal = joint_probs.sum(dim=-2)  # [..., NUM_LENGTH_BINS]
    
    return (length_marginal * length_centers).sum(dim=-1)


# dim=0 means "operate along the row axis" -> collapses rows -> one value per column
# dim=1 means "operate along the column axis" -> collapses columns -> one value per row


# === Decoding Functions===


# mask the value head logits

def mask_reserved_tokens(logits: torch.Tensor, distribution_token_id: int = DEFAULT_DISTRIBUTION_TOKEN_ID, num_reward_states: int = 2) -> torch.Tensor:

    num_bins = get_num_bins(num_reward_states)
    logits[..., distribution_token_id : distribution_token_id + num_bins] = -1e4
    return logits

