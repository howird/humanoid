# Detailed Overview of PHCPolicy Class

The `PHCPolicy` class is a neural network policy implementation for controlling humanoid characters in physics-based environments. It inherits from `DiscriminatorPolicy` and implements a policy for Physics-based Humanoid Control (PHC).

## Architecture Overview

`PHCPolicy` implements an actor-critic architecture with:
- An actor network that outputs action distributions
- A critic network that estimates value functions
- An optional discriminator for adversarial motion imitation (defined in the parent class)

## Training Pipeline

### 1. Initialization
When a `PHCPolicy` is instantiated:
- It initializes with environment information and a hidden size (default 512)
- Sets up observation normalization (`RunningNorm`)
- Configures actor, critic, and discriminator networks
- Sets action bounds based on the environment's action space

**Key Parameters:**
- `input_size`: Dimension of observation space (environment-dependent)
- `action_size`: Dimension of action space (environment-dependent)
- `soft_bound`: Action bounds (0.9 * env.single_action_space.high[0])
- `sigma`: Log standard deviation parameter (initialized to -2.9)

### 2. Observation Processing
**Function:** `encode_observations(obs)`

**Input:**
- `obs`: Raw observations from environment [batch_size, input_size]

**Process:**
1. Normalize observations using `obs_norm` (running mean/std normalization)
2. Store normalized observations in `self.obs_pointer` for critic use
3. Pass normalized observations through actor MLP

**Actor MLP Architecture:**
```
Sequential(
  Linear(input_size → 2048)
  SiLU()
  Linear(2048 → 1536)
  SiLU()
  Linear(1536 → 1024)
  SiLU()
  Linear(1024 → 1024)
  SiLU()
  Linear(1024 → 512)
  SiLU()
  Linear(512 → hidden_size)
  LayerNorm(hidden_size)
  SiLU()
)
```

**Output:**
- `hidden`: Encoded observations [batch_size, hidden_size]
- `lookup`: None (used in other policy implementations)

### 3. Action Generation
**Function:** `decode_actions(hidden, lookup=None)`

**Input:**
- `hidden`: Encoded observations [batch_size, hidden_size]
- `lookup`: Not used in this implementation

**Process:**
1. Generate action means (`mu`) using a linear layer: `hidden → action_size`
2. Use fixed standard deviation (`sigma`) expanded to match mean shape
3. Create Normal distribution with `mu` and `std`
4. If in training mode, calculate mean bound loss to penalize actions outside bounds
5. Process normalized observations through critic network to get value estimates

**Action Mean Generation:**
```
mu = Linear(hidden_size → action_size)
```

**Critic MLP Architecture:**
```
Sequential(
  Linear(input_size → 2048)
  SiLU()
  Linear(2048 → 1536)
  SiLU()
  Linear(1536 → 1024)
  SiLU()
  Linear(1024 → 1024)
  SiLU()
  Linear(1024 → 512)
  SiLU()
  Linear(512 → hidden_size)
  LayerNorm(hidden_size)
  SiLU()
  Linear(hidden_size → 1)
)
```

**Output:**
- `probs`: Normal distribution object with parameters [batch_size, action_size]
- `value`: Value estimates [batch_size, 1]

### 4. Action Sampling and Execution
During inference or rollout:
1. Sample actions from the Normal distribution
2. If `_deterministic_action` is True, the standard deviation is clamped to a small value (effectively making it deterministic)
3. Actions are executed in the environment

### 5. Discriminator (Optional, from parent class)
**Function:** `discriminate(amp_obs)`

Used for adversarial motion imitation when `use_amp_obs` is True:
1. Normalize AMP observations
2. Process through discriminator MLP
3. Output logits for real/fake classification

**Discriminator Architecture:**
```
_disc_mlp = Sequential(
  Linear(amp_obs_size → 1024)
  ReLU()
  Linear(1024 → hidden_size)
  ReLU()
)
_disc_logits = Linear(hidden_size → 1)
```

### 6. Observation Statistics Update
**Function:** `update_obs_rms(obs)` and `update_amp_obs_rms(amp_obs)`

During training:
1. Update running statistics (mean, variance) of observations
2. Update running statistics of AMP observations (if used)

### 7. Bound Loss Calculation
**Function:** `bound_loss(mu)`

During training:
1. Calculate penalty for actions outside the soft bounds
2. Used as a regularization term in the overall loss

## Tensor Shapes Throughout Pipeline

1. **Input Observation**: [batch_size, input_size]
2. **Normalized Observation**: [batch_size, input_size]
3. **Actor Hidden**: [batch_size, hidden_size]
4. **Action Mean (mu)**: [batch_size, action_size]
5. **Action Std (std)**: [batch_size, action_size]
6. **Value Estimate**: [batch_size, 1]
7. **AMP Observations** (if used): [batch_size, amp_obs_size]
8. **Discriminator Output** (if used): [batch_size, 1]

The policy uses SiLU (Sigmoid Linear Unit) activation functions throughout most of the network, with LayerNorm applied before the final activation in both actor and critic networks. This architecture is designed to handle complex humanoid control tasks with high-dimensional observation and action spaces.

> Tokens: 6.4k sent, 1.2k received. Cost: $0.04 message, $0.04 session.  
