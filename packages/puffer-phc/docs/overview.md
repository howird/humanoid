# Detailed Overview of the Training Pipeline

## Training Pipeline in `puffer_phc/clean_pufferl/__init__.py`

The training pipeline in this codebase implements a PPO (Proximal Policy
Optimization) algorithm for training humanoid characters to perform
physics-based motion imitation. Here's a detailed breakdown of the pipeline:

### 1. Initialization (`create` function)

**Purpose**: Sets up all components needed for training.

**Key Components**:

- **Experience Buffer**: Stores trajectories collected from environment
  interactions
- **Policy**: Neural network that maps observations to actions
- **Optimizer**: Updates policy parameters based on gradients
- **Profile**: Tracks performance metrics
- **TrainingState**: Maintains training progress and statistics

**Outputs**:

- `components`: Contains all training components (policy, optimizer, etc.)
- `state`: Contains training state (step count, statistics, etc.)
- `utilization`: Tracks system resource usage

### 2. Data Collection (`evaluate` function)

**Purpose**: Collects experience data by running the policy in the environment.

**Inputs**:

- `components`: Training components
- `state`: Current training state

**Process**:

1. **Environment Interaction**:
   - Receive observations `o` from environment (shape: [batch_size, obs_dim])
   - Forward observations through policy to get actions
   - If using LSTM:
     - Reset hidden states for done/truncated environments
     - Pass hidden states (h, c) to policy (shape: [num_layers, batch_size,
       hidden_size])
     - Get actions, log probabilities, values, and updated hidden states
   - Otherwise:
     - Get actions, log probabilities, and values directly
   - Send actions to environment (shape: [batch_size, action_dim])
   - Receive rewards, dones, truncations, and info

2. **Experience Storage**:
   - Store observations, values, actions, log probabilities, rewards, dones,
     truncations
   - Update global step counter
   - Collect environment statistics from info dictionaries

**Outputs**:

- Updated `state.stats`: Contains collected environment statistics
- `infos`: Raw information from environment steps

### 3. Policy Update (`train` function)

**Purpose**: Updates the policy based on collected experience.

**Inputs**:

- `components`: Training components
- `state`: Current training state

**Process**:

1. **Prepare Training Data**:
   - Sort and flatten experience data
   - Compute Generalized Advantage Estimation (GAE)
   - If using adversarial imitation learning:
     - Compute adversarial rewards using discriminator
     - Add to environment rewards

2. **Policy Optimization Loop**:
   - For each epoch and minibatch:
     - Forward observations through policy to get new action distributions and
       values
     - Compute policy loss (clipped surrogate objective)
     - Compute value function loss
     - Compute entropy bonus
     - If using discriminator:
       - Compute discriminator loss (binary classification between agent and
         demo)
     - Apply regularization (L2 to initial parameters)
     - Backpropagate gradients
     - Clip gradients to prevent exploding gradients
     - Update policy parameters

3. **Learning Rate Annealing**:
   - Optionally decay learning rate based on progress

4. **Logging and Checkpointing**:
   - Compute explained variance of value function
   - Log statistics to wandb if enabled
   - Print dashboard with training progress
   - Save checkpoint if interval reached

**Tensor Shapes**:

- Observations: [minibatch_size, obs_dim]
- Actions: [minibatch_size, action_dim]
- Log probabilities: [minibatch_size]
- Values: [minibatch_size]
- Advantages: [minibatch_size]
- Returns: [minibatch_size]
- LSTM states (if used): [num_layers, minibatch_size, hidden_size]

### 4. Cleanup (`close` function)

**Purpose**: Properly closes environment and logs final artifacts.

**Process**:

- Close vector environment
- Stop utilization tracking
- Save final model checkpoint
- Finish wandb logging if enabled

## Training Pipeline in `scripts/train.py`

This script orchestrates the overall training process, including periodic
evaluation and motion resampling.

### 1. Initialization

**Purpose**: Sets up environment, policy, and training components.

**Process**:

- Create vector environment
- Initialize policy
- Load checkpoint if provided
- Set up wandb logging if enabled
- Initialize training components, state, and utilization

### 2. Training Loop

**Purpose**: Manages the training process, including periodic evaluations and
motion resampling.

**Process**:

1. **Main Training Loop**:
   - Continue until reaching total timesteps
   - Periodically evaluate policy performance
   - Periodically resample motions to maintain diversity
   - Reset environment and LSTM states after resampling
   - Collect experience data using `clean_pufferl.evaluate`
   - Update observation normalization statistics
   - Update policy using `clean_pufferl.train`
   - Apply learning rate decay

2. **Motion Resampling**:
   - Every `motion_resample_interval` epochs:
     - Evaluate policy on all motions
     - Resample motions based on success/failure
     - Reset environment and LSTM states

3. **Evaluation**:
   - Every `checkpoint_interval` epochs:
     - Create `EvalStats` to track performance
     - Run policy on all motions
     - Compute metrics (success rate, MPJPE, etc.)
     - Log results to wandb

### 3. Final Evaluation

**Purpose**: Evaluates final policy performance.

**Process**:

- Create `EvalStats` to track performance
- Run policy on all motions with deterministic actions
- Compute comprehensive metrics
- Log final results to wandb

### 4. Cleanup

**Purpose**: Properly closes environment and logs final artifacts.

**Process**:

- Close training components
- Save final model checkpoint
- Finish wandb logging

## Key Data Structures and Their Flow

1. **Observations**:
   - From environment: [batch_size, obs_dim]
   - Stored in experience buffer
   - Normalized using running statistics

2. **Actions**:
   - Generated by policy: [batch_size, action_dim]
   - Sent to environment
   - Stored in experience buffer

3. **Rewards**:
   - From environment: [batch_size]
   - Combined with adversarial rewards if using imitation
   - Used to compute advantages

4. **Advantages**:
   - Computed using GAE: [batch_size]
   - Used for policy gradient updates

5. **LSTM States** (if using RNN):
   - Hidden and cell states: [num_layers, batch_size, hidden_size]
   - Maintained across environment steps
   - Reset for done/truncated environments

6. **Statistics**:
   - Collected in `StatsData` class
   - Includes episode returns, lengths, and task-specific metrics
   - Logged to wandb and printed to dashboard

This pipeline implements a sophisticated PPO algorithm with several extensions:

- LSTM-based policies for handling temporal dependencies
- Adversarial imitation learning for motion imitation
- Motion resampling for curriculum learning
- Comprehensive evaluation metrics for tracking performance
