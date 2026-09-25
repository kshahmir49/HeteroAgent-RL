# HeteroAgent-RL

A portfolio project for learning and demonstrating practical LLM-agent orchestration, reinforcement learning, evaluation, and inference engineering.

## Phase 1

Phase 1 builds a deterministic three-agent pipeline

Task → Planner → Executor → Verifier → Final answer

The RL controller comes later. First, we need a clean environment with stable interfaces, logging, reproducible prompts, and tests.

## Free local LLM setup

The default backend is **Ollama**, so the project does not require a paid API.

For a 16 GB Apple Silicon Mac, the current default model is

```text
qwen3:4b-instruct
```

Install Ollama from its official macOS distribution, start Ollama, then pull the model

```bash
ollama pull qwen3:4b-instruct
```

The model download is roughly 2.5 GB. The project uses a 4K context by default to keep memory usage modest.

Create a Python environment and install the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the first real local three-agent task

```bash
python -m heteroagent_rl.cli \
  --task "Write a Python function that returns the nth Fibonacci number."
```

No API key is needed for the default Ollama backend.

You can verify the pipeline without loading a real model

```bash
python -m heteroagent_rl.cli --mock \
  --task "Write a Python function that returns the nth Fibonacci number."
```

## Architecture

The current fixed baseline is

```text
Task
  ↓
Planner
  ↓
Executor
  ↓
Verifier
  ↓
Final answer
```

All three roles currently share the same local model but use different system prompts. This gives us a clean homogeneous baseline before we introduce true model heterogeneity later.

The same agent interface will later be controlled by an RL policy whose actions are

- CALL_PLANNER
- CALL_EXECUTOR
- CALL_VERIFIER
- STOP

Because the fixed workflow and the learned workflow use the same agent objects, we can compare them fairly.

## Local configuration

The default Ollama endpoint is

```text
http://localhost:11434
```

Override it with

```bash
export HETEROAGENT_OLLAMA_URL="http://localhost:11434"
```

The default context length is 4096 tokens. Override it with

```bash
export HETEROAGENT_NUM_CTX=8192
```

or per run

```bash
python -m heteroagent_rl.cli \
  --num-ctx 8192 \
  --task "Solve this problem..."
```

## Optional paid/OpenAI-compatible backend

The OpenAI client remains optional so we can compare local and hosted inference later. It is not required for the project.

Install it only if you want it

```bash
pip install -e ".[openai]"
```

Then set the relevant environment variables and choose the backend explicitly

```bash
export HETEROAGENT_API_KEY="..."
export HETEROAGENT_BASE_URL="https://api.openai.com/v1"

python -m heteroagent_rl.cli \
  --backend openai \
  --planner-model "YOUR_MODEL" \
  --executor-model "YOUR_MODEL" \
  --verifier-model "YOUR_MODEL" \
  --task "Solve this problem..."
```

## Phase 1 acceptance criteria

Before moving to RL, the project should be able to

- run the same task through three role-specialized agents
- swap models without changing orchestration code
- record every prompt, response, latency, and token count
- produce a machine-readable trajectory
- run without an external API using the mock client
- run fully locally using Ollama
- pass unit tests

## Planned heterogeneous setup

Once the environment and evaluator are stable, we will replace the homogeneous baseline with a small heterogeneous team, for example

```text
Planner    → Qwen general-purpose model
Executor   → Qwen Coder model
Verifier   → lightweight Gemma model
```

The exact model sizes will be chosen to stay within a 16 GB unified-memory Mac and a modest disk budget.

## Next milestones

### Phase 1A
Add a real coding benchmark adapter and executable scoring.

### Phase 1B
Give the Executor a sandboxed Python tool so it can run code, inspect errors, and revise solutions.

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


## Phase 1B benchmark harness

The `phase1b-eval` branch adds lightweight generation and evaluation support for EvalPlus coding tasks.

Install the benchmark dependency

```bash
pip install -e ".[bench,dev]"
```

Generate solutions for the first 5 HumanEval+ tasks without executing model-generated code

```bash
python -m heteroagent_rl.evaluate \
  --benchmark humaneval \
  --limit 5
```

This writes

```text
runs/humaneval/samples.jsonl
runs/humaneval/trajectories.jsonl
runs/humaneval/summary.json
```

The summary includes token usage, latency, and the LLM Verifier's PASS rate.

### Important safety note

By default, the benchmark command does **not** execute generated Python.

Passing `--run-tests` invokes EvalPlus correctness checks and therefore executes model-generated code locally. EvalPlus recommends sandboxing untrusted code. Do not use `--run-tests` on your host machine unless you explicitly accept that risk.

For now, use generation-only mode. We will add a safer sandboxed execution path before relying on objective pass/fail scoring.


## Phase 1C isolated execution

The evaluator can now run EvalPlus inside a container boundary instead of executing model-generated Python directly in the host process.

Check which sandbox backends are available

```bash
python -m heteroagent_rl.sandbox
```

The supported backends are

- Apple `container`
- Docker

On Apple silicon Macs running macOS 26, Apple's `container` CLI is the preferred lightweight option. Start its service with

```bash
container system start
```

Then run a small end-to-end benchmark with objective tests

```bash
python -m heteroagent_rl.evaluate \
  --benchmark humaneval \
  --limit 3 \
  --run-tests
```

The evaluator automatically prefers Apple `container` when available and falls back to Docker.

The sandboxed worker runs with

- networking disabled
- a read-only root filesystem
- read-only benchmark inputs
- a temporary writable `/tmp`
- CPU and memory limits
- only a disposable output directory mounted writable

Generated code is still untrusted. Containerization substantially improves isolation but should not be treated as a formal security proof.

The resulting summary adds objective metrics such as

```json
{
  "tests_executed": true,
  "sandbox_backend": "apple",
  "base_pass_rate": 0.0,
  "plus_pass_rate": 0.0,
  "verifier_false_pass_rate": 0.0
}
```

These metrics give us the first machine-checkable reward signal for the future RL environment.


## Phase 1E deterministic repair

The evaluator now applies a narrow deterministic repair layer after preflight and before sandbox execution.

Currently supported repair

- missing imports from `typing` when every unresolved annotation name maps unambiguously to `typing`

The evaluator preserves both versions

```text
raw_model_solution
system_solution
```

and records

```text
raw_preflight_pass_rate
preflight_pass_rate
repair_rate
```

This keeps model quality separate from system quality. A repaired solution is never reported as if the model generated it correctly on the first try.

The repair layer deliberately does not modify syntax errors, unknown unresolved names, or logic errors. Those remain visible for later tool feedback and LLM repair experiments.


## Phase 1G public-example repair

The evaluator can now use only prompt-visible examples as development feedback before held-out EvalPlus scoring.

Enable one-shot repair with

```bash
python -m heteroagent_rl.evaluate \
  --benchmark humaneval \
  --limit 25 \
  --public-repair \
  --run-tests
```

The flow is

```text
Planner
  ↓
Executor
  ↓
deterministic preflight repair
  ↓
public examples in isolated sandbox
  ↓ fail
one Repairer call
  ↓
public examples rerun
  ↓
held-out EvalPlus evaluation
```

The Repairer receives only the original task, the current candidate, and failures from examples already visible in the benchmark prompt. Held-out EvalPlus inputs are never fed back into the model.

New metrics include

```text
public_examples_initial_pass_rate
llm_repair_rate
llm_repair_success_rate
public_examples_final_pass_rate
```

Verifier calibration excludes tasks changed by the LLM Repairer because the original Verifier did not judge the repaired solution.

