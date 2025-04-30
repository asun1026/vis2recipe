import torch
import torch.nn as nn
import pickle
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# from args import get_parser

# # =============================================================================
# parser = get_parser()
# opts = parser.parse_args()
# # # =============================================================================

class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()
        
    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)


class ingRNN(nn.Module):
    # Modify __init__ to accept paths to the new files via opts
    def __init__(self, opts): # Pass opts object
        super(ingRNN, self).__init__()
        self.irnn = nn.LSTM(input_size=opts.ingrW2VDim, hidden_size=opts.irnnDim, bidirectional=True, batch_first=True)

        # --- Load pre-processed vocab and vectors ---
        print(f"Loading pre-processed vocab from: {opts.ingrVocab}") # New arg needed
        with open(opts.ingrVocab, 'rb') as f:
            keys = pickle.load(f)

        print(f"Loading pre-processed vectors from: {opts.ingrVectors}") # New arg needed
        emb_mat = torch.load(opts.ingrVectors) # Load the torch tensor directly
        # --- End loading pre-processed files ---

        # --- Initialize nn.Embedding using the loaded matrix ---
        # NOTE: Corrected the original code's error using 'vec' instead of 'emb_mat'
        # Also ensure opts.ingrW2VDim matches the loaded vector dimension
        if emb_mat.shape[1] != opts.ingrW2VDim:
             raise ValueError(f"Word2Vec dimension mismatch! Opts expected {opts.ingrW2VDim}, "
                              f"but loaded vectors have dimension {emb_mat.shape[1]}")

        self.embs = nn.Embedding(emb_mat.size(0), opts.ingrW2VDim, padding_idx=0)
        self.embs.weight.data.copy_(emb_mat)
        print(f"Initialized ingredient embedding layer with {emb_mat.size(0)} embeddings.")
        # --- End nn.Embedding initialization ---

    def forward(self, x, sq_lengths):
        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x) # This part remains the same

        # ... rest of forward method remains the same ...
        # (pack_padded_sequence, lstm, pad_packed_sequence, gather hidden states)
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        out, hidden = self.irnn(packed_seq)
        _, original_idx = sorted_idx.sort(0, descending=False)
        unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
        output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
        output = output.view(output.size(0),output.size(1)*output.size(2))
        return output

# Im2recipe model (modify __init__ to pass opts to RNNs)
class im2recipe(nn.Module):
    # Modify __init__ to accept opts
    def __init__(self, opts): # Pass opts object
        super(im2recipe, self).__init__()
        # This part likely depends on opts.preModel still
        if hasattr(opts, 'preModel') and opts.preModel=='resNet50':
            # Use modern weights API if possible
            try:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except TypeError: # Fallback for older torchvision
                resnet = models.resnet50(pretrained=True)

            modules = list(resnet.children())[:-1]
            self.visionMLP = nn.Sequential(*modules)

            # Ensure imfeatDim and embDim are in opts
            if not hasattr(opts, 'imfeatDim'): opts.imfeatDim = 2048
            if not hasattr(opts, 'embDim'): raise ValueError("opts.embDim is required")

            self.visual_embedding = nn.Sequential(
                nn.Linear(opts.imfeatDim, opts.embDim),
                nn.Tanh(),
            )

            # Check recipe embedding dims are present in opts
            if not hasattr(opts, 'irnnDim'): raise ValueError("opts.irnnDim is required")
            if not hasattr(opts, 'srnnDim'): raise ValueError("opts.srnnDim is required")
            recipe_emb_in_dim = opts.irnnDim*2 + opts.srnnDim # Bi-directional ingRNN + stRNN
            # Original code had potential typo: nn.Linear(in, out, out)? Should be nn.Linear(in, out)
            self.recipe_embedding = nn.Sequential(
                nn.Linear(recipe_emb_in_dim, opts.embDim),
                nn.Tanh(),
            )
        else:
            preModel_name = getattr(opts, 'preModel', 'Not specified')
            # Be more flexible or specific depending on needs
            raise Exception(f'Only resNet50 preModel supported currently, got: {preModel_name}')

        # Pass opts to sub-module constructors
        self.stRNN_     = stRNN(opts) # Assuming stRNN also needs opts
        self.ingRNN_    = ingRNN(opts) # Pass opts here
        self.table      = TableModule() # Doesn't need opts

        # Handle semantic regularization if needed
        self.semantic_branch = None
        if hasattr(opts, 'semantic_reg') and opts.semantic_reg:
             if not hasattr(opts, 'numClasses'): raise ValueError("opts.numClasses required for semantic_reg")
             self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, x, y1, y2, z1, z2):
        # recipe embedding (Pass inputs to submodules)
        instr_emb = self.stRNN_(y1, y2) # Assuming y1=instr features, y2=instr lengths
        ingr_emb = self.ingRNN_(z1, z2) # Assuming z1=ingr indices, z2=ingr lengths
        recipe_emb = self.table([instr_emb, ingr_emb], 1) # Concatenate features
        recipe_emb = self.recipe_embedding(recipe_emb)
        recipe_emb = norm(recipe_emb)

        # visual embedding
        visual_emb = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)

        # Handle output based on semantic regularization
        if self.semantic_branch is not None:
            visual_sem = self.semantic_branch(visual_emb)
            recipe_sem = self.semantic_branch(recipe_emb)
            output = [visual_emb, recipe_emb, visual_sem, recipe_sem]
        else:
            output = [visual_emb, recipe_emb]
        return output

# --- Need to modify stRNN too if it uses opts ---
class stRNN(nn.Module):
     # Modify __init__ to accept opts
    def __init__(self, opts): # Pass opts object
        super(stRNN, self).__init__()
        # Check instruction dims are present in opts
        if not hasattr(opts, 'stDim'): raise ValueError("opts.stDim is required")
        if not hasattr(opts, 'srnnDim'): raise ValueError("opts.srnnDim is required")
        self.lstm = nn.LSTM(input_size=opts.stDim, hidden_size=opts.srnnDim, bidirectional=False, batch_first=True)

    def forward(self, x, sq_lengths):
        # ... forward logic remains the same ...
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        out, hidden = self.lstm(packed_seq)
        _, original_idx = sorted_idx.sort(0, descending=False)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unsorted_idx = original_idx.view(-1,1,1).expand_as(unpacked)
        idx = (sq_lengths-1).view(-1,1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        output = unpacked.gather(0, unsorted_idx.long()).gather(1,idx.long())
        output = output.view(output.size(0),output.size(1)*output.size(2))
        return output
