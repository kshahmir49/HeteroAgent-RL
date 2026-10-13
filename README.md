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



## Phase 2A RL environment

Phase 2A introduces a framework-independent environment for learning when to call each agent.

The discrete action space is

```text
CALL_PLANNER
CALL_EXECUTOR
CALL_VERIFIER
CALL_REPAIR
STOP
```

The observation intentionally exposes compact orchestration signals rather than held-out test data. It includes whether a plan or candidate exists, static preflight status, public-example status, verifier verdict, call count, token use, latency, step count, and remaining budget.

Candidate inspection after Executor or Repairer output may apply the deterministic preflight repair and may run a caller-provided public-feedback function. The held-out terminal scorer is separate and is called only when the controller chooses STOP or exhausts its step budget.

The default reward is

```text
terminal correctness
- per-call penalty
- token penalty
- latency penalty
- invalid-action penalty
```

This keeps correctness as the main objective while making unnecessary model calls costly.

The environment itself does not depend on Gymnasium, Stable-Baselines3, or a specific RL algorithm. That keeps the transition logic testable before adding PPO or another learner.



## Phase 2B deterministic controller baselines

Before training a learned policy, the RL environment now supports deterministic controller baselines.

Available controllers

```text
executor_only
planner_executor
public_repair
full_pipeline
```

Run one controller on the same 25-task HumanEval slice

```bash
python -m heteroagent_rl.rl.evaluate_baselines \
  --benchmark humaneval \
  --limit 25 \
  --controller executor_only
```

Then repeat with the other controller names.

Each run uses the same RL environment and writes

```text
runs/controllers/humaneval/<controller>/trajectories.jsonl
runs/controllers/humaneval/<controller>/summary.json
```

The summary reports objective base and strict EvalPlus+ pass rates together with model-call count, token usage, latency, and policy reward.

Public examples are available as environment feedback after candidate generation, but held-out EvalPlus scoring is performed only after the episode has terminated. This preserves the separation between development feedback and terminal ground truth.

These controller baselines establish the comparison points needed before training PPO or another learned policy.


## Phase 2C explicit tools and benchmark splits

Public-example execution is now an explicit controller action rather than free feedback.

The action space is

```text
CALL_PLANNER
CALL_EXECUTOR
RUN_PUBLIC_TESTS
CALL_VERIFIER
CALL_REPAIR
STOP
```

The environment tracks LLM and sandbox-tool cost separately

```text
llm_calls
total_tokens
total_latency_s
tool_calls
total_tool_latency_s
```

Reward configuration now includes both LLM-call costs and tool-call costs. A candidate change clears stale public-test feedback, so a controller must explicitly rerun public tests after a repair.

Phase 2C also adds deterministic benchmark partitions. The default split seed is

```text
heteroagent-rl-v1
```

with 60 percent train, 20 percent development, and 20 percent test partitions. Splits are deterministic and disjoint for a given benchmark and seed.

For example

```bash
python -m heteroagent_rl.rl.evaluate_baselines \
  --benchmark humaneval \
  --split dev \
  --controller full_pipeline
```

Use `--split train` for controller training and policy development, `--split dev` for model selection and reward tuning, and `--split test` only for final held-out reporting. `--split all` remains available for regression experiments.

The baseline summaries now report tool calls and tool latency in addition to LLM calls, tokens, LLM latency, correctness, and policy reward.


## Phase 2C reliability

Long controller evaluations now checkpoint every completed episode before moving to the next task.

A run directory stores

```text
run_config.json
episodes.checkpoint.jsonl
summary.json
trajectories.jsonl
```

If a run stops because the sandbox service temporarily becomes unavailable, rerun the same command. Completed task IDs are loaded from the checkpoint and skipped automatically.

The sandbox backend availability check also retries transient failures before aborting.

For example

```bash
python -m heteroagent_rl.rl.evaluate_baselines \
  --benchmark humaneval \
  --split dev \
  --controller full_pipeline
```

If it stops after task 10, run the same command again. The evaluator resumes from the first unfinished task.

Use `--fresh` only when you intentionally want to discard the existing checkpoint and restart that split/controller from the beginning.

Split-specific outputs are stored under

```text
runs/controllers/<benchmark>/<split>/<controller>/
```

The progress summary is updated after each completed task. Held-out EvalPlus scoring still runs only after every episode has been generated, so interrupted generation does not leak terminal test information into the controller.


## Phase 2D task-aware controller state

The RL observation now includes cheap static features extracted from the task before any LLM call.

Examples include

```text
normalized prompt length
normalized line and word count
function count
helper-function count
import count
public-example count
argument count
annotation count
varargs and kwargs flags
prompt parse status
```

These features make different coding tasks distinguishable to a learned policy without exposing held-out test results.

The environment also uses the static public-example count in its action mask. `RUN_PUBLIC_TESTS` is unavailable when the benchmark prompt contains no visible examples. The deterministic public-repair and full-pipeline controllers follow the same rule, avoiding sandbox calls that cannot return useful example feedback.

Task features are recorded in each trajectory so learned-controller experiments can be inspected and reproduced.

This remains deliberately lightweight. Phase 2D does not add embeddings or another model call. The first learned policy can therefore test whether inexpensive structural task features are already enough to learn useful routing behavior.


## Phase 2E clean routing baselines

HumanEval tasks 0 through 24 are now reserved as legacy development data because they were repeatedly inspected during earlier phases.

The clean HumanEval pool is therefore

```text
HumanEval/25 through HumanEval/163
139 tasks total
```

With the default clean split configuration, that pool is partitioned deterministically into

```text
train   90 tasks
dev     24 tasks
test    25 tasks
```

The default clean split seed is

```text
heteroagent-rl-clean-v1
```

Use `--split legacy_dev` only for regression checks against the previously inspected HumanEval tasks. The clean `train`, `dev`, and `test` splits never contain HumanEval/0 through HumanEval/24.

Phase 2E also adds an executor-first escalation baseline

```text
executor_public_repair
```

Its deterministic policy is

```text
Executor
  ↓
public examples, only when prompt-visible examples exist
  ↓ fail
one Repairer call
  ↓
public examples rerun
  ↓
STOP
```

This baseline deliberately omits the Planner and Verifier. It tests whether direct generation plus evidence-triggered repair can recover failures without paying for the full pipeline.

Run it on the clean development split with

```bash
python -m heteroagent_rl.rl.evaluate_baselines \
  --benchmark humaneval \
  --split dev \
  --controller executor_public_repair \
  --fresh
```

Do not use the clean test split during policy development or reward tuning.


## Phase 2F heterogeneous specialist executor

Phase 2F adds a second generation model as an explicit orchestration action.

The default cheap Executor remains

```text
qwen3:4b-instruct
```

The default coding specialist is

```text
qwen2.5-coder:7b-instruct
```

The specialist is intentionally larger and code-focused. On Ollama the 7B Q4 model is about 4.7 GB, which keeps the experiment practical on a 16 GB Apple Silicon Mac while making the model meaningfully different from the default 4B general-purpose Executor.

Install the specialist model once

```bash
ollama pull qwen2.5-coder:7b-instruct
```

The RL action space now also includes

```text
CALL_SPECIALIST
```

Calling the specialist produces a new standalone candidate. If a cheaper candidate already exists, the specialist receives it only as an optional draft and may replace it completely. Candidate changes clear stale verifier and public-test feedback just like other generation actions.

A new deterministic baseline is available

```text
specialist_only
```

Run it on the clean HumanEval development split with

```bash
python -m heteroagent_rl.rl.evaluate_baselines \
  --benchmark humaneval \
  --split dev \
  --controller specialist_only \
  --fresh
```

The evaluator accepts `--specialist-model` if another local coding model should be tested.

The immediate Phase 2F question is not whether the specialist beats the cheap Executor on average. The important question is whether their successes are complementary. If the specialist solves tasks that the cheap Executor misses, the oracle union creates real routing headroom for a learned controller.


## Phase 2G contextual learned router

Phase 2G turns the measured Executor-versus-Specialist complementarity into a learned routing policy.

The router uses only the static task features extracted before any LLM call. It does not see held-out EvalPlus results, generated candidates, verifier feedback, or public-test feedback.

Training data is built by pairing the cached clean training trajectories from

```text
executor_only
specialist_only
```

For each task, the preferred action is the model with the higher recorded policy reward. This means correctness dominates the label, while token and latency costs break ties when both models have the same correctness outcome.

The first learned policy is deliberately small and inspectable. It is a shallow decision tree trained with training-only cross-validation over a fixed grid of tree depth and minimum leaf size. Cross-validation selects the configuration with the highest mean routed policy reward, then strict correctness is used as the next tie breaker.

Install the router training dependency

```bash
pip install -e ".[bench,dev,router]"
```

Train the router from the cached clean 90-task training trajectories

```bash
python -m heteroagent_rl.rl.train_router
```

The default output is

```text
runs/router/humaneval/contextual_tree.json
```

The serialized policy contains only the feature schema, compact decision-tree nodes, and training metadata. Inference does not require scikit-learn.

Before making new model calls, score the frozen router against the already cached 24-task clean development trajectories

```bash
python -m heteroagent_rl.rl.score_router
```

This produces

```text
runs/router/humaneval/dev_score.json
```

For an end-to-end run through the real RL environment, use

```bash
python -m heteroagent_rl.rl.evaluate_baselines \
  --benchmark humaneval \
  --split dev \
  --controller learned_router \
  --router-policy runs/router/humaneval/contextual_tree.json \
  --fresh
```

The evaluator records the router policy path and SHA-256 hash so a reported run can be tied to the exact frozen policy.

The clean test split remains untouched during Phase 2G. The project moves to final test evaluation only after the router design and training procedure are frozen.
