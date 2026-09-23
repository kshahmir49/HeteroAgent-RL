# HeteroAgent-RL

A portfolio project for learning and demonstrating practical LLM-agent orchestration, reinforcement learning, evaluation, and inference engineering.

## Phase 1

Phase 1 builds a deterministic three-agent pipeline

Task → Planner → Executor → Verifier → Final answer

The RL controller comes later. First, we need a clean environment with stable interfaces, logging, reproducible prompts, and tests.

## Why this structure

The same agent interface will later be controlled by an RL policy whose actions are

- CALL_PLANNER
- CALL_EXECUTOR
- CALL_VERIFIER
- STOP

Because the fixed workflow and the learned workflow use the same agent objects, we can compare them fairly.

## Quick start

Create a virtual environment and install the package

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the zero-cost mock pipeline

```bash
python -m heteroagent_rl.cli --mock \
  --task "Write a Python function that returns the nth Fibonacci number."
```

Run against an OpenAI-compatible endpoint

```bash
export HETEROAGENT_API_KEY="..."
export HETEROAGENT_BASE_URL="https://api.openai.com/v1"

python -m heteroagent_rl.cli \
  --task "Write a Python function that returns the nth Fibonacci number." \
  --planner-model "YOUR_MODEL" \
  --executor-model "YOUR_MODEL" \
  --verifier-model "YOUR_MODEL"
```

The same client works with a local vLLM server because vLLM exposes an OpenAI-compatible API.

## Phase 1 acceptance criteria

Before moving to RL, the project should be able to

- run the same task through three role-specialized agents
- swap models without changing orchestration code
- record every prompt, response, latency, and token count
- produce a machine-readable trajectory
- run without an external API using the mock client
- pass unit tests

## Next milestones

### Phase 1A
Add a real benchmark adapter for a small coding or reasoning dataset.

### Phase 1B
Add executable tools for the Executor, beginning with sandboxed Python execution.

### Phase 1C
Add an evaluator that records success, total tokens, total calls, latency, and cost.

### Phase 2
Wrap the workflow as an RL environment.

State will contain task information, current candidate answer, verifier feedback, call history, and remaining budget.

Action space will begin with

```text
CALL_PLANNER
CALL_EXECUTOR
CALL_VERIFIER
STOP
```

### Phase 3
Train a controller, initially with a simple policy-gradient or PPO baseline.

### Phase 4
Use heterogeneous models and compare learned orchestration against fixed workflows and always-use-the-largest-model baselines.

## Engineering principle

Do not add RL until the environment, logging, and evaluation are trustworthy. A good RL controller cannot rescue a poorly specified environment.
