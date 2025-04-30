import os
import json 
import math
import pickle
import lmdb 
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn 
from PIL import Image  
from tqdm import tqdm  
import torchvision.transforms as T  
import torchvision.models as models  
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoConfig,
    get_cosine_schedule_with_warmup 
)
from torchvision.utils import make_grid  # Utility to create a grid of images
from torchvision.transforms.functional import to_pil_image # Utility to convert tensor to PIL image

import wandb  # Weights & Biases for experiment tracking and visualization

# Set environment variable to disable tokenizer parallelism warnings/errors
# Useful when using DataLoaders with multiple workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def build_subset(layers_json='layer1.json', lmdb_dir='test_lmdb', img_root='test', subset_size=20000):
    """
    Constructs a subset of image-recipe pairs from JSON recipe data, LMDB entries, and image files.

    Args:
        layers_json (str): Path to the JSON file containing recipe details (like title, ingredients, instructions).
        lmdb_dir (str): Path to the LMDB directory containing mappings from recipe IDs to image data.
        img_root (str): Root directory where images are stored, structured by image ID components.
        subset_size (int): The desired maximum number of items in the subset.

    Returns:
        str: The file path to the created pickle file containing the subset list [(img_path, text)].
    """
    # Load recipe data from the JSON file
    with open(layers_json, 'r') as f:
        recipes = json.load(f)
        # Create a dictionary mapping recipe IDs to recipe details for quick lookup
        id2recipe = {r['id']: r for r in recipes}

    # Open the LMDB database in read-only mode, without locking (suitable for multiple readers)
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)

    subset = []  # Initialize an empty list to store the (image_path, text) pairs
    count = 0    # Counter for the number of items added to the subset

    # Start a read transaction with the LMDB database
    with env.begin() as txn:
        # Iterate through all key-value pairs in the database using a cursor
        for key, val in txn.cursor():
            # Stop if the desired subset size is reached
            if count >= subset_size:
                break

            # Decode the key (recipe ID) from bytes to string (using 'latin1' as specified by the data source)
            rid = key.decode('latin1')
            # Deserialize the value (contains image info) from bytes using pickle
            data = pickle.loads(val, encoding='latin1')

            # Extract the primary image ID associated with this recipe ID
            # Assumes the first image in the 'imgs' list is the relevant one
            img_id = data['imgs'][0]['id']
            # Construct the image path based on a specific directory structure
            # e.g., img_id = "a/b/c/d/abcd123.jpg" -> parts = ['a', 'b', 'c', 'd']
            parts = [img_id[i] for i in range(4)]
            img_path = os.path.join(img_root, *parts, img_id)

            # Check if the image file actually exists at the constructed path
            if not os.path.exists(img_path):
                continue # Skip this entry if the image is missing

            # Retrieve recipe details using the recipe ID (rid)
            rec = id2recipe.get(rid, {}) # Use .get() to handle cases where rid might not be in the JSON
            # Extract title, ingredients, and instructions, providing defaults if missing
            title = rec.get('title', '')
            ings = [i.get('text', '') for i in rec.get('ingredients', [])]
            instrs = [i.get('text', '') for i in rec.get('instructions', [])]

            # Format the extracted text into a single string
            text = f"Title: {title}\n\nIngredients:\n" + "\n".join(ings) + \
                   "\n\nInstructions:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(instrs)])

            # Append the (image path, formatted text) tuple to the subset list
            subset.append((img_path, text))
            count += 1 # Increment the counter

    # Define the output pickle file name based on the subset size
    out_pkl = f'combined_training_data_{subset_size}.pkl'
    # Save the generated subset list to the pickle file
    with open(out_pkl, 'wb') as f:
        pickle.dump(subset, f)

    # Print confirmation message
    print(f"Built subset of {len(subset)} items → {out_pkl}")
    # Return the path to the created pickle file
    return out_pkl

# ───────────────────────────────────────────────
# 1.  Image backbone  ➜  Extracts image features
# ───────────────────────────────────────────────
def build_efficientnet(device: torch.device):
    """
    Builds and prepares a pre-trained EfficientNet-B0 model for feature extraction.

    Args:
        device (torch.device): The device (CPU or CUDA) to load the model onto.

    Returns:
        tuple: A tuple containing:
            - extract_tokens (callable): A function that takes a batch of images and returns spatial features.
            - img_emb_dim (int): The dimensionality of the image embeddings produced by the feature extractor.
    """
    # Load the EfficientNet-B0 model with pre-trained weights from ImageNet
    effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Set the model to evaluation mode (disables dropout, batch norm uses running stats)
    # Move the model to the specified device (GPU or CPU)
    effnet.eval().to(device)
    # Freeze the parameters of the EfficientNet model so they are not updated during training
    effnet.requires_grad_(False)

    # Define the embedding dimension based on EfficientNet-B0's final feature map channels
    img_emb_dim = 1280  # Number of output channels before the final classifier in EfficientNet-B0

    def extract_tokens(x: torch.Tensor):
        """
        Extracts spatial feature tokens from a batch of images using the EfficientNet backbone.

        Args:
            x (torch.Tensor): Input batch of images, shape [B, 3, H, W] (e.g., [B, 3, 224, 224]).

        Returns:
            torch.Tensor: Spatial feature tokens, shape [B, H'*W', C], where C=img_emb_dim (e.g., [B, 49, 1280]).
        """
        # Pass the input images through the feature extractor part of EfficientNet
        # Note: We use effnet.features, which excludes the final pooling and classification layers.
        feats = effnet.features(x) # Output shape e.g., [B, 1280, 7, 7] for a 224x224 input

        # Get the shape components: Batch size (B), Channels (C), Height (H), Width (W)
        b, c, h, w = feats.shape
        # Reshape the features:
        # 1. Flatten the spatial dimensions (H, W) into a single dimension (H*W) -> [B, C, H*W]
        # 2. Permute dimensions to put the channel dimension last -> [B, H*W, C]
        # This results in a sequence of spatial feature vectors for each image in the batch.
        return feats.flatten(2).permute(0, 2, 1)  # Shape: [B, 49, 1280]

    # Return the feature extraction function and the embedding dimension
    return extract_tokens, img_emb_dim

# ───────────────────────────────────────────────
# 2.  Adapter: Connects Vision to Language Model
# ───────────────────────────────────────────────
class SpatialPrefixAdapter(nn.Module):
    """
    A neural network module that maps spatial image features (from EfficientNet)
    to a sequence of *k* prefix embedding tokens compatible with a Language Model (LLM).
    This acts as a bridge between the vision backbone and the LLM.
    """

    def __init__(
        self,
        img_emb_dim: int = 1280,    # Dimension of input image embeddings (from EfficientNet)
        lm_emb_dim: int = 4096,     # Dimension of the LLM's embeddings
        hidden_dim: int = 512,      # Intermediate hidden dimension within the adapter
        num_layers: int = 2,        # Number of Transformer Encoder layers in the adapter
        num_prefix_tokens: int = 4, # Number of prefix tokens (k) to generate
        dropout: float = 0.1,       # Dropout rate for regularization
    ):
        super().__init__() # Initialize the parent nn.Module class
        self.num_prefix_tokens = num_prefix_tokens # Store the number of prefix tokens

        # --- Network Layers ---
        # 1. Input Projection: Linear layer to project image embeddings to the adapter's hidden dimension.
        self.input_proj = nn.Linear(img_emb_dim, hidden_dim)

        # 2. Transformer Encoder: Processes the projected spatial tokens.
        #    - Define a single Transformer Encoder Layer.
        #    - norm_first=True applies LayerNorm before self-attention/FFN (common in modern transformers).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,       # The dimension of the input/output features for this layer
            nhead=8,                  # Number of attention heads
            dim_feedforward=hidden_dim * 4, # Dimension of the feed-forward network
            batch_first=True,         # Input/output tensors have batch dimension first (B, Seq, Dim)
            dropout=dropout,          # Dropout rate within the encoder layer
            norm_first=True,          # Apply normalization before attention/feedforward layers
        )
        #    - Stack multiple encoder layers.
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 3. Output Projection:
        #    - We don't use pooling here, instead average pooling is done in forward pass.
        #    - Linear layer to project the pooled/averaged features to the final embedding dimension.
        #      The target dimension is k * lm_emb_dim, as we need k tokens each of size lm_emb_dim.
        self.out_proj = nn.Linear(hidden_dim, lm_emb_dim * num_prefix_tokens)

        # 4. Layer Normalization: Applied to the final output prefix embeddings for stability.
        self.layernorm = nn.LayerNorm(lm_emb_dim)

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the adapter.

        Args:
            spatial_tokens (torch.Tensor): Input spatial features from the image backbone,
                                           shape [B, N, img_emb_dim] (e.g., [B, 49, 1280]).

        Returns:
            torch.Tensor: Output prefix embeddings, shape [B, k, lm_emb_dim],
                          where k = num_prefix_tokens.
        """
        # spatial_tokens shape: [B, N, img_emb_dim] (e.g., B, 49, 1280)
        # 1. Project input image embeddings to the hidden dimension
        x = self.input_proj(spatial_tokens)  # Shape: [B, N, H] (e.g., B, 49, 512)

        # 2. Pass through the Transformer Encoder
        x = self.encoder(x)  # Shape remains [B, N, H]

        # 3. Mean Pooling: Average the features across the spatial sequence dimension (N).
        #    This aggregates the spatial information into a single vector per image.
        x = x.mean(dim=1)  # Shape: [B, H] (e.g., B, 512)

        # 4. Project to the final output dimension (k * LLM embedding dimension)
        x = self.out_proj(x)  # Shape: [B, k*D] where D = lm_emb_dim (e.g., B, 4*4096)

        # 5. Reshape the output into k separate prefix tokens for each item in the batch.
        b = x.size(0) # Batch size
        k = self.num_prefix_tokens # Number of prefix tokens
        d = x.size(1) // k # LLM embedding dimension (inferred)
        x = x.view(b, k, d)  # Reshape to [B, k, D] (e.g., B, 4, 4096)

        # 6. Apply Layer Normalization to the prefix embeddings
        x = self.layernorm(x) # Shape remains [B, k, D]

        # Return the final prefix embeddings
        return x

# ───────────────────────────────────────────────
# 3.  Dataset returns raw image tensor + text
# ───────────────────────────────────────────────
class LLMTrainingDataset(Dataset):
    """
    PyTorch Dataset class for loading image-text pairs from a pre-processed pickle file.
    It handles image loading, transformation, text tokenization, and label creation.
    """
    def __init__(self, pkl_path, tokenizer, img_transform, max_len=256):
        """
        Initializes the dataset.

        Args:
            pkl_path (str): Path to the pickle file created by `build_subset`, containing [(img_path, text)] list.
            tokenizer: The Hugging Face tokenizer instance for processing text.
            img_transform: The torchvision transform pipeline to apply to images.
            max_len (int): Maximum sequence length for tokenization (including padding/truncation).
        """
        # Load the list of (image_path, text) pairs from the pickle file
        self.items = pickle.load(open(pkl_path, "rb"))
        # Store the tokenizer, image transformer, and max sequence length
        self.tok = tokenizer
        self.tr = img_transform
        self.max_len = max_len

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.items)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single item (image-text pair) from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing processed data:
                - "img": Transformed image tensor [C, H, W].
                - "input_ids": Tokenized text input IDs [L].
                - "attention_mask": Attention mask for the tokenized text [L].
                - "labels": Labels for language modeling (input_ids with padding replaced by -100) [L].
        """
        # Get the image path and corresponding text for the given index
        img_path, text = self.items[idx]

        # Load the image using PIL, ensure it's in RGB format
        img = Image.open(img_path).convert("RGB")
        # Apply the image transformations (e.g., resize, crop, normalize)
        img = self.tr(img)  # Resulting tensor shape e.g., [3, 224, 224]

        # Tokenize the text using the provided tokenizer
        toks = self.tok(
            text,                        # Input text string
            padding="max_length",        # Pad sequences to max_len
            truncation=True,             # Truncate sequences longer than max_len
            max_length=self.max_len,     # Maximum sequence length
            return_tensors="pt"          # Return PyTorch tensors
        )

        # Extract input IDs and attention mask, remove the leading batch dimension (squeeze(0))
        ids = toks.input_ids.squeeze(0)     # Shape: [max_len]
        mask = toks.attention_mask.squeeze(0) # Shape: [max_len]

        # Create labels for causal language modeling.
        # Initially, labels are the same as input_ids.
        labels = ids.clone()
        # Replace padding token IDs in labels with -100.
        # This is the standard way to tell the loss function (CrossEntropyLoss) to ignore these tokens.
        labels[labels == self.tok.pad_token_id] = -100

        # Return the processed data as a dictionary
        return {"img": img, "input_ids": ids, "attention_mask": mask, "labels": labels}

def train(
    llm_name: str,               # Name/path of the pre-trained LLM (e.g., from Hugging Face Hub)
    subset_pkl: str,             # Path to the dataset pickle file created by build_subset
    *,                           # Force subsequent arguments to be keyword-only
    epochs: int = 5,             # Number of training epochs
    bs: int = 8,                 # Batch size per device
    lr: float = 5e-4,            # Learning rate for the adapter's optimizer
    num_prefix_tokens: int = 4,  # Number of prefix tokens generated by the adapter
    grad_accum_steps: int | None = None, # Gradient accumulation steps (simulates larger batch size)
    project: str | None = None,  # Weights & Biases project name
    run_name: str | None = None, # Weights & Biases run name
):
    """
    Main training function to train the SpatialPrefixAdapter.

    Args:
        llm_name (str): Identifier for the pre-trained LLM.
        subset_pkl (str): Path to the training data pickle file.
        epochs (int): Number of training epochs.
        bs (int): Batch size.
        lr (float): Learning rate.
        num_prefix_tokens (int): Number of image prefix tokens.
        grad_accum_steps (int | None): Steps for gradient accumulation. If None, calculated for effective bs ~32.
        project (str | None): W&B project name. Uses env var or default if None.
        run_name (str | None): W&B run name. Auto-generated if None.
    """
    # Set the device to CUDA (GPU) if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------- W&B Initialization -------------
    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project=project or os.getenv("WANDB_PROJECT", "recipe-llm"), # Project name (user-provided, env var, or default)
        name=run_name, # Run name (user-provided or auto-generated by W&B)
        config=dict( # Log hyperparameters and configuration settings
            epochs=epochs,
            batch_size=bs,
            lr=lr,
            llm_name=llm_name,
            prefix_tokens=num_prefix_tokens,
            grad_accum_steps=grad_accum_steps, # Log calculated value later if None initially
            dataset=subset_pkl,
        ),
    )

    # ------------- LLM (4‑bit, frozen) -------------
    print(f"Loading LLM: {llm_name}...")
    # Configure BitsAndBytes for 4-bit quantization
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,                 # Enable 4-bit loading
        bnb_4bit_quant_type="nf4",         # Use NF4 quantization type (recommended)
        bnb_4bit_compute_dtype=torch.float16 # Set compute dtype to float16 for faster computation
    )
    # Load the pre-trained Causal Language Model
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        quantization_config=bnb_cfg, # Apply the 4-bit quantization configuration
        device_map={"": 0}           # Automatically map model layers to available devices (e.g., GPU 0)
                                     # "{"" : 0}" is shorthand for loading the whole model on device 0
    ).eval() # Set the LLM to evaluation mode - its weights will be frozen.
    print("LLM loaded.")

    # Load the tokenizer associated with the LLM
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True) # trust_remote_code needed for some models
    # Set the padding token to be the same as the end-of-sentence token if not already set
    # Important for causal LMs where padding is often handled differently.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    # ------------- Vision backbone + adapter -------------
    print("Setting up vision backbone and adapter...")
    # Build the EfficientNet feature extractor
    extract_tokens, img_emb_dim = build_efficientnet(device)
    # Instantiate the SpatialPrefixAdapter
    adapter = SpatialPrefixAdapter(
        img_emb_dim=img_emb_dim,                # Input dimension from EfficientNet
        lm_emb_dim=llm.config.hidden_size,      # Output dimension matching LLM's hidden size
        num_prefix_tokens=num_prefix_tokens,    # Number of prefix tokens to generate
        # Using default hidden_dim, num_layers, dropout
    ).to(device) # Move the adapter to the specified device
    print(f"Adapter created with {num_prefix_tokens} prefix tokens.")
    # Print number of trainable parameters (should only be the adapter's)
    num_trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Number of trainable parameters (Adapter): {num_trainable_params:,}")

    # ------------- Dataset -------------
    print(f"Loading dataset from: {subset_pkl}...")
    # Define the image transformations for training
    # Includes data augmentation like random cropping and color jittering
    transform = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.8, 1.0)), # Randomly resize and crop to 224x224
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), # Random color adjustments
            T.ToTensor(), # Convert PIL Image to PyTorch tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize using ImageNet stats
        ]
    )
    # Create the training dataset instance
    ds = LLMTrainingDataset(subset_pkl, tokenizer, transform, max_len=llm.config.max_position_embeddings // 2) # Use a reasonable max_len
    print(f"Dataset loaded with {len(ds)} items.")

    # Create the DataLoader for batching, shuffling, and parallel loading
    loader = DataLoader(
        ds,
        batch_size=bs,        # Batch size per iteration
        shuffle=True,         # Shuffle data at the beginning of each epoch
        num_workers=4,        # Number of worker processes for data loading (adjust based on system)
        pin_memory=True       # Speeds up data transfer to GPU if True
    )

    # ------------- Optimizer & scheduler -------------
    # Initialize the AdamW optimizer, targeting only the adapter's parameters
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

    # Calculate gradient accumulation steps if not provided
    # Aims for an effective batch size of roughly 32 (bs * grad_accum_steps ≈ 32)
    if grad_accum_steps is None:
        grad_accum_steps = max(1, 32 // bs)
    wandb.config.update({"grad_accum_steps": grad_accum_steps}) # Update W&B config if calculated
    print(f"Using batch size: {bs}, grad_accum_steps: {grad_accum_steps} (Effective BS: {bs * grad_accum_steps})")

    # Calculate total training steps and warmup steps for the scheduler
    # total_steps accounts for gradient accumulation
    total_steps = math.ceil(len(loader) / grad_accum_steps) * epochs
    warmup_steps = int(0.02 * total_steps) # Use 2% of total steps for warmup
    print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

    # Create a cosine learning rate scheduler with warmup
    sched = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Watch the adapter model with W&B to track gradients and topology
    wandb.watch(adapter, log="gradients", log_freq=100) # Log gradients every 100 steps

    # ------------- Constant prompt -------------
    # Define a fixed text prompt that will precede the LLM's generated recipe
    PROMPT = "Generate a recipe for the pictured dish:\n"
    # Tokenize the prompt (without adding special tokens like BOS/EOS here, as they might be handled later or by the model)
    # `add_special_tokens=False` might be safer depending on the tokenizer/model. Adjust if needed.
    prompt_ids = tokenizer(PROMPT, add_special_tokens=False).input_ids
    # Convert prompt IDs to a tensor and move to the device
    prompt_ids = torch.tensor(prompt_ids, device=device)
    prompt_len = prompt_ids.size(0) # Store the length of the prompt tokens
    print(f"Using fixed prompt: '{PROMPT}' (Length: {prompt_len} tokens)")

    # ------------- Training Loop -------------
    print("Starting training...")
    adapter.train() # Set the adapter to training mode (enables dropout, etc.)
    step_global = 0 # Initialize global step counter (tracks optimizer steps)

    # Loop over the specified number of epochs
    for ep in range(epochs):
        # Wrap the dataloader with tqdm for a progress bar
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
        # Loop over batches in the current epoch
        for step_local, batch in enumerate(pbar):

            # ---------------- Forward pass ----------------
            # 1. Extract image features and generate prefix embeddings
            imgs = batch["img"].to(device)          # Move images to device -> [B, 3, H, W]
            spatial = extract_tokens(imgs)        # Get spatial features -> [B, N, C_img]
            prefix_emb = adapter(spatial)         # Generate prefix embeddings -> [B, k, D_llm]

            # 2. Prepare text inputs (token IDs, attention mask, labels)
            ids = batch["input_ids"].to(device)     # Move token IDs to device -> [B, L]
            mask = batch["attention_mask"].to(device) # Move attention mask to device -> [B, L]
            labels = batch["labels"].to(device)     # Move labels to device -> [B, L]

            B, L = ids.shape # Batch size and sequence length of text data
            k = num_prefix_tokens # Number of prefix tokens

            # 3. Prepend the constant text prompt to the recipe text
            #    Expand the prompt_ids tensor to match the batch size
            p_ids = prompt_ids.unsqueeze(0).expand(B, -1) # -> [B, P] where P=prompt_len
            #    Concatenate prompt IDs with recipe IDs along the sequence dimension
            ids = torch.cat([p_ids, ids], dim=1) # -> [B, P+L]
            #    Create a mask for the prompt (all ones) and concatenate with recipe mask
            mask = torch.cat([torch.ones_like(p_ids), mask], dim=1) # -> [B, P+L]
            #    Create labels for the prompt (all -100, ignored by loss) and concatenate
            labels = torch.cat([torch.full_like(p_ids, -100), labels], dim=1) # -> [B, P+L]

            # -------------- Embedding Concatenation --------------
            # 4. Get the standard word embeddings for the combined text (prompt + recipe)
            #    Note: We access the LLM's embedding layer directly.
            base_emb = llm.get_input_embeddings()(ids)  # -> [B, P+L, D_llm]

            # 5. Ensure prefix embeddings have the same data type as base embeddings (e.g., float16 if using mixed precision)
            prefix_emb = prefix_emb.to(base_emb.dtype)

            # 6. Concatenate the image prefix embeddings with the text embeddings
            #    Prefix comes first: [Image Prefix | Prompt | Recipe Text]
            all_emb = torch.cat([prefix_emb, base_emb], dim=1)  # -> [B, k+P+L, D_llm]

            # Ensure concatenated embeddings match the required dtype for the LLM (if needed, usually handled by device mapping/quantization)
            # all_emb = all_emb.to(base_emb.dtype) # Redundant if prefix_emb already converted

            # 7. Update attention mask and labels to account for the added prefix tokens
            #    Create a mask for the prefix (all ones, as they should be attended to)
            prefix_mask = torch.ones((B, k), dtype=mask.dtype, device=device) # -> [B, k]
            #    Create labels for the prefix (all -100, as LLM doesn't predict these)
            prefix_lbl = torch.full((B, k), -100, dtype=labels.dtype, device=device) # -> [B, k]
            #    Concatenate prefix mask/labels with the existing text mask/labels
            mask = torch.cat([prefix_mask, mask], dim=1)     # -> [B, k+P+L]
            labels = torch.cat([prefix_lbl, labels], dim=1) # -> [B, k+P+L]

            # ---------------- LLM Forward & Loss Calculation ----------------
            # 8. Pass the combined embeddings and attention mask to the frozen LLM
            #    The LLM computes logits and the loss based on the provided labels.
            outputs = llm(
                inputs_embeds=all_emb,  # Pass embeddings directly, not input_ids
                attention_mask=mask,    # Provide the combined attention mask
                labels=labels           # Provide the combined labels for loss calculation
            )
            # 9. Get the loss value from the model output.
            #    Divide the loss by gradient accumulation steps for proper scaling.
            loss = outputs.loss / grad_accum_steps

            # ---------------- Backward pass & Optimization Step ----------------
            # 10. Compute gradients for the adapter parameters based on the scaled loss.
            loss.backward()

            # 11. Perform optimizer step, scheduler step, and zero gradients
            #     only after accumulating gradients for `grad_accum_steps`.
            if (step_local + 1) % grad_accum_steps == 0 or (step_local + 1) == len(loader):
                # Clip gradients to prevent exploding gradients (max norm of 1.0)
                # Applied only to adapter parameters as they are the only ones being optimized.
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)

                # Update adapter weights using the optimizer
                opt.step()
                # Update learning rate using the scheduler
                sched.step()
                # Reset gradients for the next accumulation cycle
                # set_to_none=True can improve performance slightly.
                opt.zero_grad(set_to_none=True)

                # Increment the global step counter (counts optimizer steps)
                step_global += 1

                # Log metrics to W&B periodically (e.g., every 50 global steps)
                if step_global % 50 == 0:
                    wandb.log(
                        {
                            # Log the unscaled loss (loss.item() * grad_accum_steps)
                            "train/loss": loss.item() * grad_accum_steps,
                            "lr": sched.get_last_lr()[0], # Log current learning rate
                            "epoch": ep,
                            "step": step_global,
                        }
                    )

            # Update the progress bar description with the current unscaled loss
            pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")
            # --- End of Batch Loop ---

        # Print status at the end of each epoch
        print(f"Epoch {ep+1} finished. Last loss = {loss.item() * grad_accum_steps:.4f}")
        # --- End of Epoch Loop ---

    # ------------- Save Model & Finish -------------
    # Save the trained adapter's state dictionary
    adapter_save_path = "image_prefix_adapter.pth"
    torch.save(adapter.state_dict(), adapter_save_path)
    print(f"Adapter weights saved to {adapter_save_path}")

    # Finish the W&B run
    wandb.finish()
    print("Training complete. W&B run finished.")


# ───────────────────────────────────────────────
# 5.  Inference demo (generates recipe from image)
# ───────────────────────────────────────────────
@torch.no_grad() # Decorator to disable gradient calculations during inference
def demo(
    img_path: str,              # Path to the input image
    adapter_ckpt: str = "image_prefix_adapter.pth", # Path to the trained adapter checkpoint
    llm_name: str = "mistralai/Mistral-7B-Instruct-v0.3", # Name of the LLM to use
    num_prefix_tokens: int = 4, # Number of prefix tokens used during training
):
    """
    Runs inference to generate a recipe for a given image using the trained adapter and LLM.

    Args:
        img_path (str): Path to the input image file.
        adapter_ckpt (str): Path to the saved adapter weights (.pth file).
        llm_name (str): Identifier for the pre-trained LLM.
        num_prefix_tokens (int): The number of prefix tokens the adapter was trained with.
    """
    print("Starting inference demo...")
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------- Load Models -------------
    print(f"Loading LLM: {llm_name}...")
    # Load the LLM (consider using quantization if memory is an issue, but not strictly necessary for demo)
    # Using device_map for potential multi-GPU or efficient loading.
    llm = AutoModelForCausalLM.from_pretrained(llm_name, device_map={"": 0}).eval()
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    # Set padding token if necessary
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("LLM and Tokenizer loaded.")

    print("Loading vision backbone and adapter...")
    # Build the EfficientNet feature extractor
    extract_tokens, img_emb_dim = build_efficientnet(device)
    # Instantiate the Adapter
    adapter = SpatialPrefixAdapter(
        img_emb_dim=img_emb_dim,
        lm_emb_dim=llm.config.hidden_size,
        num_prefix_tokens=num_prefix_tokens,
    ).to(device)
    # Load the trained weights into the adapter
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
    # Set the adapter to evaluation mode
    adapter.eval()
    print(f"Adapter loaded from {adapter_ckpt}.")

    # ------------- Image Preprocessing -------------
    # Define image transformations suitable for inference (no random augmentation)
    tr = T.Compose(
        [
            T.Resize(256),       # Resize shorter side to 256 pixels
            T.CenterCrop(224),   # Crop the center 224x224 pixels
            T.ToTensor(),        # Convert to tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize
        ]
    )
    # Load, process the image, add a batch dimension (unsqueeze(0)), and move to device
    print(f"Processing image: {img_path}")
    img = tr(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device) # Shape: [1, 3, 224, 224]

    # ------------- Feature Extraction & Prefix Generation -------------
    # Extract spatial features from the image
    spatial = extract_tokens(img) # Shape: [1, N, C_img]
    # Generate prefix embeddings using the trained adapter
    prefix_emb = adapter(spatial)  # Shape: [1, k, D_llm]

    # ------------- Prepare Inputs for LLM Generation -------------
    # Define the text prompt
    prompt = "Generate a recipe for the pictured dish:\n"
    # Tokenize the prompt and move to device
    toks = tokenizer(prompt, return_tensors="pt").to(device) # Contains 'input_ids' and 'attention_mask'

    # Get the base embeddings for the prompt tokens
    base_emb = llm.get_input_embeddings()(toks.input_ids) # Shape: [1, P, D_llm]

    # Ensure prefix embeddings match the data type of base embeddings
    prefix_emb = prefix_emb.to(base_emb.dtype)

    # Concatenate prefix embeddings and prompt embeddings
    all_emb = torch.cat([prefix_emb, base_emb], dim=1) # Shape: [1, k+P, D_llm]

    # Create the attention mask for the combined sequence (prefix + prompt)
    # Prefix mask is all ones, concatenated with the prompt's attention mask
    mask = torch.cat(
        [torch.ones((1, num_prefix_tokens), dtype=toks.attention_mask.dtype, device=device), toks.attention_mask], dim=1
    ) # Shape: [1, k+P]

    # ------------- Generate Text with LLM -------------
    print("Generating recipe...")
    # Use the LLM's generate method with the combined embeddings and mask
    gen_ids = llm.generate(
        inputs_embeds=all_emb,          # Provide the concatenated embeddings
        attention_mask=mask,            # Provide the corresponding attention mask
        max_new_tokens=250,             # Maximum number of new tokens to generate after the prompt
        do_sample=True,                 # Enable sampling for more diverse outputs
        temperature=0.7,                # Controls randomness (lower = more deterministic)
        top_p=0.9,                      # Nucleus sampling threshold (considers tokens with cumulative prob > top_p)
        pad_token_id=tokenizer.pad_token_id, # Ensure padding token is set for generation
        eos_token_id=tokenizer.eos_token_id, # Ensure EOS token is set
    )

    # ------------- Decode and Print Output -------------
    # Decode the generated token IDs (including the prompt) back into text
    # skip_special_tokens=True removes tokens like <|eos|> from the output string
    generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print("\n--- Generated Recipe ---")
    print(generated_text)
    print("--- End of Generation ---")


# Main execution block
if __name__ == "__main__":
    # Step 1: Build a subset of the data (or load an existing one)
    # Adjust subset_size as needed (e.g., use a small value for quick testing)
    print("Building data subset...")
    # Creates 'combined_training_data_10000.pkl' if it doesn't exist
    subset_pkl = build_subset(subset_size=10000)
    print(f"Using dataset: {subset_pkl}")

    # Step 2: Train the adapter (Commented out by default)
    # Uncomment this block to run training. Adjust parameters as needed.
    print("Starting training process...")
    train(
        # Choose an LLM (ensure compatibility with hidden size if changing adapter defaults)
        llm_name="facebook/opt-1.3b", # Example: smaller LLM
        #llm_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # Original LLM
        subset_pkl=subset_pkl,         # Path to the dataset created above
        epochs=3,                      # Number of epochs (adjust for convergence)
        bs=8,                          # Batch size (adjust based on GPU memory)
        lr=3e-4,                       # Learning rate
        num_prefix_tokens=4,           # Must match adapter definition if not specified
        # grad_accum_steps=4,          # Optional: Set manually if needed
        project="recipe-llm-prefix",   # W&B Project name
        run_name="prefix_spatial_opt1.3b_test", # W&B Run name (change for different runs)
    )
    print("Training finished (or was skipped).")

    # # Step 3: Run the inference demo using the first image from the created subset
    # print("Running inference demo...")
    # # Load the subset to get an image path for the demo
    # # Ensure the adapter checkpoint ('image_prefix_adapter.pth') exists from a previous training run
    # adapter_checkpoint = "image_prefix_adapter.pth"
    # if os.path.exists(adapter_checkpoint):
    #     # Load the first item's image path from the subset pickle file
    #     first_img_path = pickle.load(open(subset_pkl, "rb"))[0][0]
    #     # Call the demo function
    #     demo(
    #         img_path=first_img_path,
    #         adapter_ckpt=adapter_checkpoint,
    #         # Ensure LLM name matches the one used for training or is compatible
    #         llm_name="facebook/opt-1.3b", # Example: smaller LLM
    #         # llm_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # Original LLM
    #         num_prefix_tokens=4 # Should match the trained adapter
    #     )
    # else:
    #     print(f"Adapter checkpoint '{adapter_checkpoint}' not found. Skipping demo.")
    #     print("Please train the model first by uncommenting the 'train(...)' block.")

    print("Script finished.")