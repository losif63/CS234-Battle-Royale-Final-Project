"""Neural network, observation packing, and action helpers for the battle royale PPO agent."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_AMMO_DEPOSITS, NUM_HEALTH_PICKUPS

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
MAX_AGENTS = 10
LSTM_HIDDEN = 128

# Entity packing dimensions
# own(10) + lidar(12) + global(2) = 24
SELF_DIM = 10 + 12 + 2
# Type one-hot: [bullet, ammo_deposit, health_pickup, agent] = 4
# Bullets: 4 feats padded to 8 -> + 4 one-hot = 12
# Deposits: 2 feats padded to 8 -> + 4 one-hot = 12
ENTITY_DIM = 12  # 4 type one-hot + 8 max feature dims
MAX_VISIBLE_BULLETS = 10
N_ENTITIES = MAX_VISIBLE_BULLETS + NUM_AMMO_DEPOSITS + NUM_HEALTH_PICKUPS + (MAX_AGENTS - 1)  # 10 + 12 + 15 + 9 = 46

# Discrete action space: move_x(3), move_y(3), fire(2), heal(2)
# Rotation is continuous via Beta distribution mapped to [-pi, pi]
DISCRETE_ACTION_HEADS = [3, 3, 2, 2]
NUM_DISCRETE_ACTIONS = 4


# ---------------------------------------------------------------------------
# Entity Packing
# ---------------------------------------------------------------------------

def pack_actor_obs(obs_dict):
    """Pack actor observations into (self_features, entities, entity_mask).

    Returns:
        self_features: (B, A, 23)
        entities:      (B, A, N, 11)
        entity_mask:   (B, A, N) bool — True = valid entity
    """
    own = obs_dict["own"]           # (B, A, 9)
    lidar = obs_dict["lidar"]       # (B, A, 12)
    global_obs = obs_dict["global"] # (B, A, 2)

    bullets = obs_dict["bullets"]       # (B, A, K, 4)
    bullet_mask = obs_dict["bullet_mask"]  # (B, A, K)
    deposits = obs_dict["deposits"]     # (B, A, D, 2)
    deposit_mask = obs_dict["deposit_mask"]  # (B, A, D)
    health_pickups = obs_dict["health_pickups"]     # (B, A, H, 2)
    health_pickup_mask = obs_dict["health_pickup_mask"]  # (B, A, H)
    agents = obs_dict["agents"]         # (B, A, A-1, 8)
    agent_mask = obs_dict["agent_mask"] # (B, A, A-1)

    B, A = own.shape[:2]
    K = bullets.shape[2]
    D = deposits.shape[2]
    H = health_pickups.shape[2]
    A_other = agents.shape[2]
    device = own.device

    self_features = torch.cat([own, lidar, global_obs], dim=-1)  # (B, A, 23)

    # Type one-hots: [bullet, ammo_deposit, health_pickup, agent]
    # Bullets: type=[1,0,0,0], features padded from 4 to 8
    bullet_type = torch.zeros(B, A, K, 4, device=device)
    bullet_type[..., 0] = 1.0
    bullet_pad = torch.zeros(B, A, K, 4, device=device)  # pad 4->8
    bullet_ent = torch.cat([bullet_type, bullets, bullet_pad], dim=-1)  # (B, A, K, 12)

    # Deposits: type=[0,1,0,0], features padded from 2 to 8
    deposit_type = torch.zeros(B, A, D, 4, device=device)
    deposit_type[..., 1] = 1.0
    deposit_pad = torch.zeros(B, A, D, 6, device=device)  # pad 2->8
    deposit_ent = torch.cat([deposit_type, deposits, deposit_pad], dim=-1)  # (B, A, D, 12)

    # Health pickups: type=[0,0,1,0], features padded from 2 to 8
    hp_type = torch.zeros(B, A, H, 4, device=device)
    hp_type[..., 2] = 1.0
    hp_pad = torch.zeros(B, A, H, 6, device=device)  # pad 2->8
    hp_ent = torch.cat([hp_type, health_pickups, hp_pad], dim=-1)  # (B, A, H, 12)

    # Agents: type=[0,0,0,1], features already 8
    agent_type = torch.zeros(B, A, A_other, 4, device=device)
    agent_type[..., 3] = 1.0
    agent_ent = torch.cat([agent_type, agents], dim=-1)  # (B, A, A-1, 12)

    # Concatenate all entities
    entities = torch.cat([bullet_ent, deposit_ent, hp_ent, agent_ent], dim=2)  # (B, A, N, 11)
    entity_mask = torch.cat([bullet_mask, deposit_mask, health_pickup_mask, agent_mask], dim=2)  # (B, A, N)

    return self_features, entities, entity_mask


# ---------------------------------------------------------------------------
# Attention Actor-Critic Network
# ---------------------------------------------------------------------------

class AttentionActorCritic(nn.Module):
    def __init__(self, self_dim=SELF_DIM, entity_dim=ENTITY_DIM,
                 hidden=64, n_heads=4, lstm_hidden=LSTM_HIDDEN):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.lstm_hidden = lstm_hidden

        # Shared encoders
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden), nn.ReLU(),
        )
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, hidden), nn.ReLU(),
        )

        # Cross-attention projections (manual for torch.compile compat)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)

        # Post-attention MLP (shared encoder output)
        self.post_attn = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )

        # LSTM — actor only
        self.lstm = nn.LSTMCell(128, lstm_hidden)

        # Actor head input = 128 (from skip connection: reactive + lstm)
        self.actor_mlp = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
        )

        # Discrete action heads
        self.move_x_head = nn.Linear(128, 3)
        self.move_y_head = nn.Linear(128, 3)
        self.fire_head = nn.Linear(128, 2)
        self.heal_head = nn.Linear(128, 2)

        # Continuous rotation head (Beta distribution parameters)
        self.rotate_alpha = nn.Linear(128, 1)
        self.rotate_beta = nn.Linear(128, 1)

        # Critic (attention-based, shares encoder + LSTM hidden state)
        self.critic_mlp = nn.Sequential(
            nn.Linear(128 + lstm_hidden, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        for head in [self.move_x_head, self.move_y_head, self.fire_head, self.heal_head]:
            nn.init.orthogonal_(head.weight, gain=0.01)
        # Beta heads: gain=0.01 so softplus(~0)+1 ≈ 1.69, giving near-uniform Beta
        for head in [self.rotate_alpha, self.rotate_beta]:
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_mlp[-1].weight, gain=1.0)

    def _cross_attention(self, query, keys, values, mask):
        """Manual multi-head cross-attention.

        Args:
            query:  (*, 1, hidden)
            keys:   (*, N, hidden)
            values: (*, N, hidden)
            mask:   (*, N) bool — True = attend to this entity
        Returns:
            (*, hidden)
        """
        B_flat = query.shape[0]
        H = self.n_heads
        D = self.head_dim

        # Project and reshape to (B, H, seq, D)
        q = self.q_proj(query).view(B_flat, 1, H, D).transpose(1, 2)    # (B, H, 1, D)
        k = self.k_proj(keys).view(B_flat, -1, H, D).transpose(1, 2)    # (B, H, N, D)
        v = self.v_proj(values).view(B_flat, -1, H, D).transpose(1, 2)  # (B, H, N, D)

        # Scaled dot-product attention
        scale = D ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, 1, N)

        # Apply mask: set masked positions to -inf
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        attn = attn.masked_fill(~mask_expanded, float('-inf'))

        attn = F.softmax(attn, dim=-1)  # (B, H, 1, N)
        attn = torch.nan_to_num(attn, nan=0.0)  # handle all-masked case

        out = torch.matmul(attn, v)  # (B, H, 1, D)
        out = out.transpose(1, 2).reshape(B_flat, self.hidden)  # (B, hidden)
        return out

    def _encode(self, self_feat, entities, entity_mask):
        """Encode observations through attention. Returns (*, 128)."""
        self_embed = self.self_encoder(self_feat)
        ent_embed = self.entity_encoder(entities)
        query = self_embed.unsqueeze(-2)
        attn_out = self._cross_attention(query, ent_embed, ent_embed, entity_mask)
        combined = torch.cat([self_embed, attn_out], dim=-1)
        return self.post_attn(combined)

    def forward_actor(self, self_feat, entities, entity_mask, hx=None, cx=None):
        """
        Args:
            self_feat:   (B, self_dim)
            entities:    (B, N, entity_dim)
            entity_mask: (B, N) bool
            hx, cx:      LSTM state (B, lstm_hidden) or None
        Returns:
            (discrete_logits, alpha, beta, (hx, cx))
            discrete_logits: tuple of 4 tensors (move_x, move_y, fire, heal)
            alpha, beta: (B,) Beta distribution parameters for continuous rotation
        """
        reactive = self._encode(self_feat, entities, entity_mask)  # (B, 128)

        B = reactive.shape[0]
        if hx is None:
            hx = torch.zeros(B, self.lstm_hidden, device=reactive.device)
            cx = torch.zeros(B, self.lstm_hidden, device=reactive.device)

        hx, cx = self.lstm(reactive, (hx, cx))
        features = reactive + hx  # skip connection
        h = self.actor_mlp(features)

        discrete_logits = (self.move_x_head(h), self.move_y_head(h),
                           self.fire_head(h), self.heal_head(h))
        alpha = F.softplus(self.rotate_alpha(h).squeeze(-1)) + 1.0  # (B,)
        beta = F.softplus(self.rotate_beta(h).squeeze(-1)) + 1.0    # (B,)
        return discrete_logits, alpha, beta, (hx, cx)

    def forward_critic(self, self_feat, entities, entity_mask, hx=None):
        """Critic shares the encoder + LSTM hidden state, outputs scalar value."""
        h = self._encode(self_feat, entities, entity_mask)
        if hx is None:
            hx = torch.zeros(h.shape[0], self.lstm_hidden, device=h.device)
        h = torch.cat([h, hx], dim=-1)
        return self.critic_mlp(h).squeeze(-1)

    def get_action_and_value(self, self_feat, entities, entity_mask,
                             hx=None, cx=None,
                             discrete_actions=None, continuous_actions=None):
        """
        Returns:
            (disc_actions, cont_actions, log_prob, entropy, value, hx, cx)
            disc_actions: (B, 4) long — move_x, move_y, fire, heal
            cont_actions: (B,) float — rotation in [0, 1] (Beta sample)
        """
        discrete_logits, alpha, beta_param, (hx_out, cx_out) = self.forward_actor(
            self_feat, entities, entity_mask, hx, cx)

        # Apply action masks
        discrete_logits = _apply_action_masks(discrete_logits, self_feat)

        dists = [torch.distributions.Categorical(logits=l) for l in discrete_logits]

        # Beta distribution for continuous rotation (float32 for numerical stability)
        beta_dist = torch.distributions.Beta(alpha.float(), beta_param.float())

        if discrete_actions is None:
            disc_actions = torch.stack([d.sample() for d in dists], dim=-1)
            cont_actions = beta_dist.sample()  # (B,) in [0, 1]
        else:
            disc_actions = discrete_actions
            # Clamp stored continuous actions to valid Beta range
            cont_actions = continuous_actions.float().clamp(1e-6, 1.0 - 1e-6)

        log_prob = sum(d.log_prob(disc_actions[:, i]) for i, d in enumerate(dists))
        # Beta log prob with Jacobian correction for mapping [0,1] -> [-pi, pi]
        beta_lp = beta_dist.log_prob(cont_actions.clamp(1e-6, 1.0 - 1e-6))
        log_prob = log_prob + beta_lp - math.log(2 * math.pi)

        # Separate heal entropy for per-head weighting
        base_entropy = sum(d.entropy() for i, d in enumerate(dists) if i != 3) + beta_dist.entropy()
        heal_entropy = dists[3].entropy()

        value = self.forward_critic(self_feat, entities, entity_mask, hx=hx_out)
        return disc_actions, cont_actions, log_prob, base_entropy, heal_entropy, value, hx_out, cx_out


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _actions_to_sim(discrete_actions, continuous_rotation, agent_dir):
    """Convert discrete (4-col) + continuous rotation to sim inputs.

    discrete_actions: (..., 4) — [move_x(3), move_y(3), fire(2), heal(2)]
    continuous_rotation: (...) — rotation in [0, 1], mapped to [-pi, pi]
    agent_dir: (...) — current agent facing direction
    """
    move_x = discrete_actions[..., 0].float() - 1.0   # 0→-1, 1→0, 2→+1
    move_y = discrete_actions[..., 1].float() - 1.0   # 0→-1, 1→0, 2→+1
    rotation_delta = continuous_rotation.float() * 2 * math.pi - math.pi  # [0,1] -> [-pi, pi]
    aim_angle = agent_dir + rotation_delta
    fire = discrete_actions[..., 2].bool()
    heal = discrete_actions[..., 3].bool()
    return move_x, move_y, aim_angle, fire, heal


def _apply_action_masks(discrete_logits, self_feat):
    """Apply action masks based on observations. Returns masked logits tuple.

    self_feat: (..., SELF_DIM) — self_feat[:, 4]=health/maxHP, [:, 6]=ammo/max, [:, 7]=medkits/max
    discrete_logits: (move_x, move_y, fire, heal) — 4 tensors
    """
    ammo = self_feat[..., 6]       # normalized ammo (0 = empty)
    medkits = self_feat[..., 7]    # normalized medkits (0 = none)
    health = self_feat[..., 4]     # normalized health (1.0 = full)

    move_x_logits, move_y_logits, fire_logits, heal_logits = discrete_logits

    # Mask fire[1] (yes) when no ammo
    fire_mask = (ammo <= 0).unsqueeze(-1) * torch.tensor([0.0, 1.0], device=fire_logits.device)
    fire_logits = fire_logits - fire_mask * 1e9

    # Mask heal[1] (yes) when no medkits OR full health
    heal_block = ((medkits <= 0) | (health >= 1.0)).unsqueeze(-1)
    heal_mask = heal_block * torch.tensor([0.0, 1.0], device=heal_logits.device)
    heal_logits = heal_logits - heal_mask * 1e9

    return (move_x_logits, move_y_logits, fire_logits, heal_logits)


def _sample_actions(discrete_logits, alpha, beta_param):
    """Sample discrete + continuous actions. Returns (disc (B, 4) long, cont (B,) float)."""
    disc = torch.stack(
        [torch.distributions.Categorical(logits=l).sample() for l in discrete_logits],
        dim=-1)
    cont = torch.distributions.Beta(alpha.float(), beta_param.float()).sample()
    return disc, cont


def _greedy_actions(discrete_logits, alpha, beta_param):
    """Deterministic actions. Returns (disc (*, 4) long, cont (*,) float)."""
    disc = torch.stack([l.argmax(dim=-1) for l in discrete_logits], dim=-1)
    # Beta mode: (alpha-1)/(alpha+beta-2), clamped to [0,1]
    cont = ((alpha - 1.0) / (alpha + beta_param - 2.0).clamp(min=1e-6)).clamp(0.0, 1.0)
    return disc, cont
