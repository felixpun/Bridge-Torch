import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal



class GCBCAgent(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        encode_dim: int = 512,
        hidden_dim: int =256,
        dropout_rate: float = 0.1,
        action_dim: int = 7,
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.mlp = nn.Sequential(
            nn.Linear(encode_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.SiLU(inplace=True)
        )

        self.action_mean_linear = nn.Linear(hidden_dim, action_dim)

        self.register_buffer("fixed_std", torch.eye(action_dim))

    def forward(self, obs_imgs, goal_imgs):
        observation_and_goal = torch.concat((obs_imgs, goal_imgs), dim=-3)

        outputs = self.mlp(self.encoder(observation_and_goal))

        means = self.action_mean_linear(outputs)

        dist = MultivariateNormal(means, scale_tril=self.fixed_std)
        
        return dist
    
    @torch.no_grad()
    def sample_actions(self, obs, goal_obs, argmax=True):
        obs_img = torch.tensor(obs["image"], device=self.fixed_std.device).unsqueeze_(0)
        goal_img = torch.tensor(goal_obs["image"], device=self.fixed_std.device).unsqueeze_(0)

        dist = self.forward(obs_img, goal_img)

        if argmax:
            actions = dist.mode
        else:
            actions = dist.sample()
        return actions
