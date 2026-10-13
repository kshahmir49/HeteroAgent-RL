# Experiment history

This document records the main decisions that led from the original multi-agent pipeline to the final heterogeneous router.

## Phase 1

The initial system used a fixed Planner, Executor, and Verifier pipeline. The goal was to establish stable interfaces, reproducible prompts, usage logging, and a local Ollama backend before adding learning.

Sandboxed EvalPlus evaluation was then added so generated code could be scored objectively without executing it directly in the host Python process.

A narrow deterministic repair layer fixed only unambiguous preflight issues such as missing typing imports.

Public-example repair was tested using only examples visible in benchmark prompts. Held-out EvalPlus failures were never fed back into the model.

## Phase 2A through 2D

The fixed workflow was wrapped as a framework-independent RL environment.

The action space grew to include Planner, Executor, public-test execution, Verifier, Repairer, and Stop. Reward tracked terminal correctness as well as LLM calls, tokens, latency, tool calls, and invalid actions.

Deterministic controller baselines showed that larger multi-agent pipelines often cost more without improving strict correctness.

Static task features were added to the environment so a learned policy could distinguish tasks before any LLM call.

## Phase 2E

Early HumanEval tasks had already been repeatedly inspected, so HumanEval/0 through HumanEval/24 were reserved as legacy development data.

The clean pool became HumanEval/25 through HumanEval/163, split deterministically into 90 train, 24 dev, and 25 test tasks.

An Executor-first public-repair controller was evaluated on clean dev. It matched Executor-only correctness while using more calls, tokens, and latency. This removed the main justification for training a sequential repair policy at that stage.

## Phase 2F

A second heterogeneous model was introduced.

The general Executor remained qwen3:4b-instruct and the Specialist used qwen2.5-coder:7b-instruct.

On clean dev, the two models showed complementary failures. Executor-only reached 83.33% strict correctness, Specialist-only reached 91.67%, and the two-model oracle reached 95.83%.

On the 90-task clean training split, Executor reached 78.89% strict correctness, Specialist reached 80.00%, and the two-model oracle reached 88.89%.

That complementarity created real routing headroom.

## Phase 2G

A contextual router was trained from paired clean training trajectories.

Labels were chosen by policy reward so correctness dominated while token and latency costs could break ties.

Training-only cross-validation selected a shallow decision tree with maximum depth 2 and minimum leaf size 8.

The frozen router used policy SHA-256

```text
01428874602c7f5f02dd8cd59300b5d774b3452f0d225aa806a2f9591af9bb65
```

On cached clean dev trajectories the router reached 95.83% strict correctness, exactly matching the two-model oracle.

A live end-to-end dev run reproduced the same 95.83% result.

## Final untouched test

No additional router or prompt tuning was performed after dev.

On the 25-task test split

| Method | Strict EvalPlus+ |
| --- | ---: |
| Executor only | 76% |
| Specialist only | 92% |
| Learned router | 92% |
| Two-model oracle | 96% |

The router did not beat the Specialist on the final test set. It tied the Specialist while coming within one routable task of the two-model oracle.

The project stops here by design. No repeated test reruns, post-test prompt changes, or test-driven router tuning are part of the reported system.
