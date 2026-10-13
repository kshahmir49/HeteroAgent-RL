# Experiment 2 preregistered protocol

This document freezes the Experiment 2 design before any MBPP+ results are inspected.

## Research question

Can HeteroAgent-RL preserve most of the coding quality of a substantially larger specialist model while reducing how often that specialist must be used?

Experiment 1 established that routing is learnable, but the original model pair did not create a useful efficiency tradeoff because the coding specialist was already competitive on both quality and measured inference cost.

Experiment 2 therefore uses a deliberately asymmetric model pair.

## Model pair

| Role | Ollama model | Approximate local model size |
| --- | --- | ---: |
| Cheap general Executor | `qwen3:1.7b` | 1.4 GB |
| Coding Specialist | `qwen2.5-coder:7b-instruct` | 4.7 GB |

The Executor is intentionally much smaller than the Specialist. The routing problem is therefore meaningful only if the cheap model can solve a useful fraction of tasks on its own.

The prompts, task-feature extractor, router architecture, reward weights, sandbox, and evaluation logic are otherwise inherited from Experiment 1.

## Benchmark

Experiment 2 uses MBPP+ through EvalPlus.

The installed EvalPlus release exposes 378 MBPP+ tasks. HeteroAgent-RL applies the existing deterministic split procedure with seed

```text
heteroagent-rl-mbpp-v1
```

and the existing 65 / 17.5 / 17.5 proportions.

This yields

| Split | Tasks | Use |
| --- | ---: | --- |
| Train | 245 | Generate paired trajectories and train the router |
| Dev | 66 | One round of model selection and frozen-router validation |
| Test | 67 | Final untouched evaluation |

The test split must not be run until the router design is frozen.

## Baselines

Three deployable policies are compared

- Executor only
- Specialist only
- Learned router

A two-model oracle is reported only as an upper bound. It is not deployable because it uses terminal benchmark outcomes to select the successful model after the fact.

## Router training

The same contextual routing method from Experiment 1 is reused.

The router receives only static task features available before any LLM call. It does not receive generated code, public-test results, verifier feedback, or held-out EvalPlus outcomes at inference time.

Training labels are derived only from paired train trajectories. The router is a shallow decision tree selected by training-only cross-validation.

No new semantic features, embeddings, prompt-specific rules, or benchmark-task heuristics may be added after looking at MBPP dev outcomes.

## Primary success criterion

Experiment 2 is considered a successful routing result only if the frozen learned router satisfies both conditions on the untouched MBPP+ test split

1. strict EvalPlus+ accuracy is within 3 percentage points of Specialist-only
2. the router sends no more than 60 percent of test tasks to the Specialist

This criterion directly tests the project claim. Matching specialist quality is not enough if the router still calls the Specialist almost all the time.

## Secondary efficiency metrics

The final report also includes

- total and average LLM latency
- total and average tokens
- average policy reward
- fraction of tasks routed to each model
- two-model oracle strict accuracy

Latency is the primary measured runtime-efficiency metric. Token count is reported for completeness but is not treated as a model-compute metric because the two models have different parameter counts.

## Stopping rule

The experiment stops after one frozen test evaluation.

After the test split is evaluated, there will be no changes to model choice, prompts, features, reward weights, router architecture, hyperparameters, split seed, or success criterion followed by another test run.

If the router fails the preregistered criterion, that failure is reported as the result rather than tuned away.

## Commands

Pull the cheap model if needed

```bash
ollama pull qwen3:1.7b
ollama pull qwen2.5-coder:7b-instruct
```

Generate paired training trajectories

```bash
python -m heteroagent_rl.rl.evaluate_baselines \
  --benchmark mbpp \
  --split train \
  --split-seed heteroagent-rl-mbpp-v1 \
  --controller executor_only \
  --executor-model qwen3:1.7b \
  --specialist-model qwen2.5-coder:7b-instruct \
  --fresh

python -m heteroagent_rl.rl.evaluate_baselines \
  --benchmark mbpp \
  --split train \
  --split-seed heteroagent-rl-mbpp-v1 \
  --controller specialist_only \
  --executor-model qwen3:1.7b \
  --specialist-model qwen2.5-coder:7b-instruct \
  --fresh
```

The router is trained only after both training runs are complete. Dev is evaluated only after the router has been fitted. Test remains untouched until the dev analysis is complete and the policy is frozen.
