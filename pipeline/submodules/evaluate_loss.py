import torch
import itertools
import json

from datasets import load_dataset

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase

def batch_iterator_chat_completions(dataset_instructions, dataset_outputs, tokenize_instructions_fn, batch_size, eoi_toks):
    it_instructions = iter(dataset_instructions)
    it_outputs = iter(dataset_outputs)
    while True:
        instructions_batch = list(itertools.islice(it_instructions, batch_size))
        outputs_batch = list(itertools.islice(it_outputs, batch_size))
        if not instructions_batch or not outputs_batch:
            break
        inputs = tokenize_instructions_fn(instructions=instructions_batch, outputs=outputs_batch)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        # also mask out all tokens before the eoi token region
        for b in range(inputs["input_ids"].shape[0]):
            for i in range(inputs["input_ids"].shape[1]):

                if torch.all(inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks):
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

                # normally the above condition works. but the tokenization instruction tokens in Llama2 is not clean, and so we need this hack
                if eoi_toks.shape[0] == 6 and (inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks).sum().item() >= eoi_toks.shape[0] - 2:
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

        yield inputs, loss_mask 

def batch_iterator_custom_completions(completions_file_path: str, tokenize_instructions_fn, batch_size, eoi_toks):
    """Yields batches from the custom completions."""

    custom_completions = json.load(open(completions_file_path, 'r'))

    instructions, completions = [], []

    for i in range(len(custom_completions)):
        instructions.append(custom_completions[i]['prompt'])
        completions.append(custom_completions[i]['response'])

    return batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, batch_size, eoi_toks)

def batch_iterator_alpaca(tokenize_instructions_fn, batch_size, eoi_toks):
    """Yields batches from the Alpaca dataset."""

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42)

    instructions, completions = [], []

    for i in range(len(dataset)):
        if dataset[i]['input'].strip() == '': # filter for instructions that do not have inputs
            instructions.append(dataset[i]['instruction'])
            completions.append(dataset[i]['output'])

    return batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, batch_size, eoi_toks)

def batch_iterator_pile(tokenizer, batch_size, max_length):
    """Yields batches from the Pile dataset."""
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True, trust_remote_code=True)

    it_dataset = iter(dataset)
    while True:
        batch = list(itertools.islice(it_dataset, batch_size))
        if not batch:
            break
        inputs = tokenizer([b['text'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        yield inputs, loss_mask

def compute_loss_over_dataset(model, tokenizer, batch_iterator, n_batches=256, fwd_pre_hooks=[], fwd_hooks=[]):
    accumulated_loss = torch.tensor(0, dtype=torch.float64, device=model.device)
    accumulated_n_tokens = torch.tensor(0, dtype=torch.int64, device=model.device)

    batch_idx = 0
    for inputs, loss_mask in batch_iterator:
        if n_batches != -1 and batch_idx >= n_batches:
            break

        inputs = inputs.to(model.device)
        loss_mask = loss_mask.to(model.device)

        input_ids = inputs["input_ids"]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            model_outputs = model(**inputs)

        logits = model_outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_for_labels = log_probs[:, :-1].gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # add a last column of zeros to log_probs_for_labels to match the shape of loss_mask
        log_probs_for_labels = torch.cat(
            [
                log_probs_for_labels,
                torch.zeros(log_probs_for_labels.shape[0]).unsqueeze(-1).to(log_probs_for_labels)
            ],
            dim=-1
        )

        # apply loss_mask
        log_probs_for_labels = log_probs_for_labels * loss_mask.to(log_probs_for_labels.device)

        accumulated_loss += -log_probs_for_labels.sum()
        accumulated_n_tokens += loss_mask.sum()

        batch_idx += 1
    
    ce_loss = accumulated_loss / accumulated_n_tokens
    perplexity = torch.exp(ce_loss)    

    return ce_loss, perplexity, accumulated_n_tokens

def evaluate_loss(
    model_base: ModelBase,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    batch_size=16,
    n_batches=256,
    max_seq_length=256,
    dataset_labels=["pile", "alpaca", "alpaca_custom_completions"],
    completions_file_path=None
):
    result = {}

    for label in dataset_labels:
        if label == 'pile':
            dataset_iterator = batch_iterator_pile(model_base.tokenizer, batch_size=batch_size, max_length=max_seq_length)
            n = n_batches
        elif label == 'alpaca':
            dataset_iterator = batch_iterator_alpaca(model_base.tokenize_instructions_fn, batch_size=batch_size, eoi_toks=torch.tensor(model_base.eoi_toks))
            n = n_batches
        elif label == 'alpaca_custom_completions':
            assert completions_file_path is not None, "A file path must be passed to load the completions"

            dataset_iterator = batch_iterator_custom_completions(
                completions_file_path=completions_file_path,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                batch_size=batch_size,
                eoi_toks=torch.tensor(model_base.eoi_toks)
            )
            n = -1 # process all completions
        else:
            raise ValueError(f"Unknown dataset label: {label}")

        ce_loss, perplexity, n_tokens = compute_loss_over_dataset(model_base.model, model_base.tokenizer, dataset_iterator, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, n_batches=n)
        print(f"{label.upper()} DATASET:")
        print(f"CE loss: {ce_loss.item()}, Perplexity: {perplexity.item()}, N tokens: {n_tokens.item()}")

        result[label] = {
            "ce_loss": ce_loss.item(),
            "perplexity": perplexity.item(),
            "n_tokens": n_tokens.item()
        }

    return result