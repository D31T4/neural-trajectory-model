import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml.model.modules import BaseStationEmbedding, TrajectoryEncoder

class TrajectoryModel(nn.Module):
    '''
    Implement trajectory model described in proposal
    '''

    def __init__(
        self,
        base_station_embedding: BaseStationEmbedding,
        trajectory_encoder: TrajectoryEncoder,
    ):
        '''
        Args:
        ---
        - base_station_embedding: base station embedding model
        - trajectory_encoder: trajectory encoder model
        '''
        assert base_station_embedding.out_dim == trajectory_encoder.out_dim

        super().__init__()
        
        self.base_station_embedding = base_station_embedding
        self.trajectory_encoder = trajectory_encoder

    def forward(
        self, 
        context: torch.FloatTensor, 
        trajectories: torch.FloatTensor, 
        candidates: torch.FloatTensor,
        candidates_context: torch.FloatTensor
    ) -> torch.FloatTensor:
        '''
        train forward.

        Args:
        ---
        - context: context tensor [B, L, dim_c]
        - trajectories: sequence of base station embeddings [B, L, dim_t]
        - candidates: n unique candidate base stations [n, dim_t]
        - candidates_context: [B, L, dim_c]

        Returns:
        ---
        - scaled dot-product attention [B, L, n]
        '''
        # [B, L, dim]
        trajectories = self.base_station_embedding(
            trajectories, 
            context
        )

        # [B, L, n, dim]
        candidates = self.base_station_embedding(
            candidates[None, None, :].repeat(trajectories.size(0), candidates_context.size(1), 1, 1),
            candidates_context.unsqueeze(-2).repeat(1, 1, candidates.size(-2), 1)
        )

        trajectories = self.trajectory_encoder(trajectories)
        dim = trajectories.shape[-1]

        # batch scaled dot product
        out: torch.FloatTensor = torch.matmul(trajectories[:, :, None, :], candidates.transpose(-2, -1))

        out = out.squeeze(2).contiguous()
        return out
    
    @torch.no_grad()
    def cost_matrix(
        self,
        context: torch.FloatTensor, 
        trajectories: torch.FloatTensor, 
        candidates: torch.FloatTensor,
        candidates_context: torch.FloatTensor
    ) -> torch.FloatTensor:
        '''
        compute cost matrix for inference

        Args:
        ---
        - context: [L, dim_c]
        - trajectories: [B, L, dim_t]
        - candidates: [n, dim_t]
        - candidates_context: [dim_c]
        '''
        # [B, L, dim]
        trajectories = self.base_station_embedding(
            trajectories, 
            context.unsqueeze(0).repeat(trajectories.size(0), 1, 1)
        )

        # [n, dim]
        candidates = self.base_station_embedding(
            candidates,
            candidates_context.unsqueeze(0).repeat(candidates.size(-2), 1)
        )

        trajectories = self.trajectory_encoder(trajectories)

        # [B, dim]
        trajectory_embeddigs = trajectories[:, -1]

        # batch scaled dot product
        out: torch.FloatTensor = torch.matmul(trajectory_embeddigs, candidates.transpose(-2, -1))

        return -out.log_softmax(dim=-1)

