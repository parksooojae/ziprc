import modal

app = modal.App("ziprc-train")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "wandb",
        "vllm",
    )
    .add_local_dir(".", remote_path="/root/ziprc")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=5400,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    lr: float = 1e-5,
    epochs: int = 1,
    batch_size: int = 1,
    max_length: int = 512,
    gamma: float = 0.99,
    bellman_weight: float = 0.01,
    alpha_kl: float = 10.0,
    num_rollouts: int = 4,
    max_prompts: int = 500,
):
    import sys
    sys.path.insert(0, "/root/ziprc")

    import copy
    import random
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import wandb

    from utils import (
        extract_joint_logits,
        get_num_bins,
        mask_reserved_tokens,
        compute_reward_bin_edges,
        NUM_LENGTH_BINS,
        DEFAULT_DISTRIBUTION_TOKEN_ID,
        LENGTH_BINS,
        DEFAULT_REWARD_VALUES,
    )
    from eval import reward as compute_reward
    from data.data_prep import prep_gsm8k

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = prep_gsm8k()["train"]
    num_prompts = min(max_prompts, len(dataset))

    # === Phase 1: Generate on-policy rollouts with vLLM ===
    # Paper Section 6.1: "For each prompt, we generate two on-policy rollouts
    # per model ... We then label each rollout for correctness."
    print(f"Generating {num_rollouts} on-policy rollouts per prompt ({num_prompts}/{len(dataset)} prompts) with vLLM...")
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="float16")
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=max_length,
        n=num_rollouts,
    )

    prompts = [dataset[i]["prompt"] for i in range(num_prompts)]
    answers = [dataset[i]["answer"] for i in range(num_prompts)]
    vllm_outputs = llm.generate(prompts, sampling_params)

    rollout_data = []
    num_correct = 0
    for i, output in enumerate(vllm_outputs):
        for completion in output.outputs:
            gen_text = completion.text
            r = compute_reward(gen_text, answers[i])
            num_correct += int(r == 1.0)
            rollout_data.append({
                "text": f"{prompts[i]}\n{gen_text}",
                "reward": r,
            })

    num_incorrect = len(rollout_data) - num_correct
    print(f"Generated {len(rollout_data)} rollouts ({num_correct} correct, {num_incorrect} incorrect)")

    # Free vLLM GPU memory before loading HF model for training
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

    # #region agent log — verify dtype fix
    print(f"[DBG] model dtype={next(model.parameters()).dtype}")
    # #endregion

    # === Phase 2: Train with KL regularization ===
    # Paper eq. 3: L(s_t) = L_aux(s_t) + α_KL · KL(π ∥ π_θ)
    # Frozen reference model provides π (original policy before training)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    num_bins = get_num_bins()

    run = wandb.init(
        entity="chrispark",
        project="zip-rc",
        config={
            "model_name": model_name,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "gamma": gamma,
            "bellman_weight": bellman_weight,
            "alpha_kl": alpha_kl,
            "num_rollouts": num_rollouts,
            "max_prompts": num_prompts,
            "dataset": "gsm8k",
            "num_bins": num_bins,
            "num_length_bins": NUM_LENGTH_BINS,
            "num_reward_states": 2,
            "distribution_token_id": DEFAULT_DISTRIBUTION_TOKEN_ID,
            "length_bins": LENGTH_BINS,
            "reward_values": DEFAULT_REWARD_VALUES,
            "optimizer": "AdamW",
            "gpu": "T4",
            "rollout_count": len(rollout_data),
            "rollout_correct": num_correct,
            "rollout_incorrect": num_incorrect,
        },
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    global_step = 0

    for epoch in range(epochs):
        random.shuffle(rollout_data)
        total_loss = 0.0
        num_steps = (len(rollout_data) + batch_size - 1) // batch_size

        for step in range(num_steps):
            batch_items = rollout_data[step * batch_size : (step + 1) * batch_size]
            texts = [item["text"] for item in batch_items]
            rewards = [item["reward"] for item in batch_items]

            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**enc)
            logits = outputs.logits.float()  # [B, T, V] — cast to float32 for numerical stability

            # === KL loss (batched over all positions) ===
            # Paper eq. 3: KL(π ∥ π_θ) over V \ R (non-reserved vocab)
            with torch.no_grad():
                ref_logits = ref_model(**enc).logits.float()

            cur_logits_for_kl = mask_reserved_tokens(logits.clone())
            ref_logits_for_kl = mask_reserved_tokens(ref_logits.clone())
            del ref_logits
            ref_probs = F.softmax(ref_logits_for_kl, dim=-1)
            del ref_logits_for_kl
            cur_log_probs = F.log_softmax(cur_logits_for_kl, dim=-1)
            del cur_logits_for_kl
            kl_per_token = F.kl_div(cur_log_probs, ref_probs, log_target=False, reduction="none").nan_to_num(0.0).sum(dim=-1)
            del cur_log_probs, ref_probs
            kl_loss = (kl_per_token * enc.attention_mask).sum() / enc.attention_mask.sum()

            # === Supervised + Bellman loss (vectorized) ===
            B, T, V = logits.shape
            seq_lens = enc.attention_mask.sum(dim=-1)  # [B]

            positions = torch.arange(T - 1, device=device)  # [T-1]
            valid_mask = positions.unsqueeze(0) < (seq_lens.unsqueeze(1) - 1)  # [B, T-1]
            num_positions = valid_mask.sum().item()

            if num_positions > 0:
                # tokens_remaining[b, t] = seq_lens[b] - t - 1
                tokens_remaining = seq_lens.unsqueeze(1) - positions.unsqueeze(0) - 1  # [B, T-1]
                tokens_remaining = tokens_remaining.clamp(min=0)

                # Vectorized get_joint_bin
                length_boundaries = torch.tensor(LENGTH_BINS[1:], device=device)
                length_bins = torch.bucketize(tokens_remaining, length_boundaries, right=True)
                length_bins = length_bins.clamp(max=NUM_LENGTH_BINS - 1)

                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)  # [B]
                reward_edges = compute_reward_bin_edges(DEFAULT_REWARD_VALUES)
                reward_boundaries = torch.tensor(reward_edges[1:-1], dtype=torch.float32, device=device)
                reward_bins = torch.bucketize(rewards_tensor, reward_boundaries, right=True)  # [B]

                target_bins = length_bins + reward_bins.unsqueeze(1) * NUM_LENGTH_BINS  # [B, T-1]

                # Supervised cross-entropy over joint distribution logits
                joint_logits_all = extract_joint_logits(logits[:, :-1, :])  # [B, T-1, num_bins]
                ce_per_pos = F.cross_entropy(
                    joint_logits_all.reshape(-1, joint_logits_all.shape[-1]),
                    target_bins.reshape(-1),
                    reduction='none',
                )  # [B*(T-1)]
                sup_loss = (ce_per_pos * valid_mask.reshape(-1).float()).sum() / num_positions

                # Bellman loss: MSE between joint logits at t and t+1
                logits_t = extract_joint_logits(logits[:, :-2, :])   # [B, T-2, num_bins]
                logits_t1 = extract_joint_logits(logits[:, 1:-1, :]) # [B, T-2, num_bins]

                bellman_mask = positions[:-1].unsqueeze(0) < (seq_lens.unsqueeze(1) - 2)  # [B, T-2]
                bellman_per_pos = (logits_t - logits_t1.detach()).pow(2).mean(dim=-1)  # [B, T-2]
                bellman_loss = (bellman_per_pos * bellman_mask.float()).sum() / bellman_mask.sum().clamp(min=1)
            else:
                sup_loss = 0.0
                bellman_loss = 0.0

            # Paper eq. 3: L = L_aux + α_KL · KL  (+ your bellman term)
            loss = sup_loss + bellman_weight * bellman_loss + alpha_kl * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            sup_val = sup_loss.item() if torch.is_tensor(sup_loss) else sup_loss
            bellman_val = bellman_loss.item() if torch.is_tensor(bellman_loss) else bellman_loss
            kl_val = kl_loss.item()
            total_loss += loss_val

            wandb.log({
                "train/loss": loss_val,
                "train/sup_loss": sup_val,
                "train/bellman_loss": bellman_val,
                "train/kl_loss": kl_val,
                "train/num_positions": num_positions,
                "train/seq_len": enc.attention_mask.sum(dim=-1).float().mean().item(),
                "epoch": epoch,
                "global_step": global_step,
            }, step=global_step)

            # #region agent log — verify NaN fix
            if step in (0, 1, 5, 10, 20):
                print(f"[DBG] step={step} sup={sup_val} bellman={bellman_val} kl={kl_val} loss={loss_val}")
            # #endregion

            if step % 10 == 0:
                print(
                    f"epoch {epoch} step {step}/{num_steps} loss {loss_val:.4f} "
                    f"(sup={sup_val:.4f}, bellman={bellman_val:.4f}, kl={kl_val:.6f})"
                )

            global_step += 1

        epoch_avg_loss = total_loss / num_steps
        wandb.log({"train/epoch_avg_loss": epoch_avg_loss, "epoch": epoch}, step=global_step)
        print(f"epoch {epoch} avg_loss {epoch_avg_loss:.4f}")

    wandb.finish()

    # Push to Hugging Face Hub
    hf_repo = "parksoojae/zip-rc-b"
    print(f"Pushing model to {hf_repo}...")
    model.push_to_hub(hf_repo)
    tokenizer.push_to_hub(hf_repo)
    print(f"Pushed to https://huggingface.co/{hf_repo}")

    return {"status": "complete", "epochs": epochs, "num_rollouts_generated": len(rollout_data)}


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    lr: float = 1e-5,
    epochs: int = 1,
    batch_size: int = 1,
    max_length: int = 512,
    max_prompts: int = 500,
):
    result = train.remote(
        model_name=model_name,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        max_prompts=max_prompts,
    )
    print(f"Training complete: {result}")
