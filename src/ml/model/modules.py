'''
torch modules
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Sequential):
    '''
    multi-level perceptron
    '''
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        do_prob: float = 0.
    ):
        '''
        Args:
        ---
        - in_dim: input dimension
        - out_dim: output dimension
        - do_prob: dropout probability
        '''
        super().__init__(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(do_prob)
        )

#region embedding
class BaseStationEmbedding(nn.Module):
    '''
    base station embedding
    '''
    def __init__(
        self,
        feat_dim: tuple[int, int],
        context_dim: tuple[int, int],
        out_dim: int,
        layer_norm: bool = True
    ):
        '''
        Args:
        ---
        - feat_dim: [feature input dim, feature output dim]
        - context_dim: [context input dim, context output dim]
        - out_dim: output dim
        '''
        super().__init__()

        self.out_dim = out_dim

        feat_in, feat_out = feat_dim
        context_in, context_out = context_dim

        self.feature_mixer = nn.Sequential(MLPBlock(feat_in, feat_out), nn.LayerNorm(feat_out))
        self.context_mixer = nn.Sequential(MLPBlock(context_in, context_out), nn.LayerNorm(context_out))

        self.output_mixer = MLPBlock(feat_out + context_out, out_dim)

        if layer_norm:
            self.ln = nn.LayerNorm(out_dim)
        else:
            self.ln = None

        
    def forward(self, feat: torch.FloatTensor, context: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Args:
        ---
        - feat: feature
        - context: context

        Returns:
        ---
        - base station embedding
        '''
        feat = self.feature_mixer(feat)
        context = self.context_mixer(context)

        feat = torch.cat((feat, context), dim=-1)
        feat = self.output_mixer(feat)

        if self.ln:
            feat = self.ln(feat)

        return feat

class ContextFreeBaseStationEmbedding(BaseStationEmbedding):
    '''
    base station embedding independent of context
    '''
    def __init__(
        self,
        feat_dim: tuple[int, int],
        out_dim: int,
        layer_norm: bool = True
    ):
        '''
        Args:
        ---
        - feat_dim: [feature input dim, feature output dim]
        - context_dim: [context input dim, context output dim]
        - out_dim: output dim
        '''
        nn.Module.__init__(self)

        self.out_dim = out_dim

        feat_in, feat_out = feat_dim

        self.feature_mixer = nn.Sequential(MLPBlock(feat_in, feat_out), nn.LayerNorm(feat_out))
        self.output_mixer = MLPBlock(feat_out, out_dim)

        if layer_norm:
            self.ln = nn.LayerNorm(out_dim)
        else:
            self.ln = None

    def forward(self, feat: torch.FloatTensor, context: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Args:
        ---
        - feat: feature
        - context: context

        Returns:
        ---
        - base station embedding
        '''
        feat = self.feature_mixer(feat)
        feat = self.output_mixer(feat)

        if self.ln:
            feat = self.ln(feat)

        return feat
#endregion

class PositionalEncoding(nn.Module):
    '''
    sinusoid positional encoding from Attention is All You Need.

    stolen from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    '''
    
    def __init__(self, 
        dim: int,
        max_len: int,
        do_prob: float = 0.
    ):
        '''
        Args:
        ---
        - dim: model dimension
        - max_len: max sequence length
        - do_prob: dropout probability
        '''
        super().__init__()
        
        self.dropout = nn.Dropout(p=do_prob)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Args:
        ---
        - x: [B, L, dim]

        Returns:
        ---
        - x + pe [B, L, dim]
        '''
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#region trajectory encoder
class TrajectoryEncoder(nn.Module):
    '''
    Abstract trajectory encoder
    '''
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Abstract method. Override this in your implementation.

        Args:
        ---
        - x: [B, L, d_in]

        Returns:
        ---
        - logits: [B, L + 1, d_out]
        '''
        raise NotImplementedError()

class TransformerTrajectoryEncoder(TrajectoryEncoder):
    '''
    transformer trajectory encoder

    Complexity of forward: 
    without attention cache: O(L^2)
    with attention cache: O(L)
    '''
    
    def __init__(
        self,
        in_dim: int,
        max_len: int,
        hid_dim: tuple[int, int, int] = (128, 256, 8),
        do_prob: float = 0.,
        n_blocks: int = 8
    ):
        '''
        Args:
        ---
        - in_dim: input dim
        - max_len: max sequence length
        - hid_dim: [attention dim, feed forward dim, no. of attention heads]
        - do_prob: dropout probability
        - n_blocks: no. of transformer blocks
        '''
        super().__init__(in_dim)

        attn_dim, ff_dim, n_head = hid_dim

        self.pe = PositionalEncoding(
            dim=in_dim,
            max_len=max_len + 1,
            do_prob=do_prob
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=n_head,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=do_prob
        )

        self.seq_model = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_blocks,
        )

        self.logits_layer = nn.Linear(
            in_features=attn_dim,
            out_features=in_dim
        )

        self.bos = nn.Parameter(
            torch.zeros(1, 1, in_dim)
        )


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Args:
        ---
        - x: [B, L, d_in]

        Returns:
        ---
        - logits: [B, L + 1, d_out]
        '''
        batch_size, seq_len = x.shape[:2]

        # prepend [bos] token
        x = torch.cat((self.bos.repeat(batch_size, 1, 1), x), dim=1)
        
        # positional encoding
        x = self.pe(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len + 1, device=x.device)

        # transformer blocks
        x = self.seq_model(x, causal_mask, is_causal=True)

        return self.logits_layer(x)
#endregion

if __name__ == '__main__':
    '''
    dryrun
    '''
    bs_embedding = BaseStationEmbedding(
        feat_dim=(2, 64),
        context_dim=(31, 48),
        out_dim=128
    )

    trajectory_encoder = TransformerTrajectoryEncoder(
        in_dim=128,
        out_dim=128,
        max_len=48,
        hid_dim=(128, 256, 8),
        n_blocks=8
    )

    # encode trajectory
    trajectory = torch.tensor([[
        [116.51172, 39.92123], [116.51135,39.93883], [116.51135,39.93883], [116.51627,39.91034], [116.47186,39.91248]
    ]])

    context = torch.cat((
        F.one_hot(torch.tensor([15, 15, 15, 15, 16]), 24),
        F.one_hot(torch.tensor(1), 7).repeat((5, 1))
    ), dim=-1).float().reshape((1, 5, -1))

    trajectory = bs_embedding(trajectory, context)
    trajectory = trajectory_encoder(trajectory)

    # compute negative log likelihood
    candidate_ctx = torch.cat((
        F.one_hot(torch.tensor(17), 24),
        F.one_hot(torch.tensor(1), 7)
    ), dim=-1).float().repeat((4, 1))

    candidate_bs = torch.tensor([
        [116.51172, 39.92123], [116.51135,39.93883], [116.51627,39.91034], [116.47186,39.91248]
    ])

    candidate_bs = bs_embedding(candidate_bs, candidate_ctx)

    logp: torch.FloatTensor = -F.log_softmax(trajectory[:, -1] @ torch.transpose(candidate_bs, 0, 1), dim=-1)
    print(logp)