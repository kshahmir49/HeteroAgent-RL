# Final results

The final reported benchmark is HumanEval+ using the clean deterministic split seed `heteroagent-rl-clean-v1`.

## Final test split

The untouched test split contains 25 tasks.

| Method | Base pass | Strict EvalPlus+ | Tokens | LLM latency | Avg policy reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Executor only | 84% | 76% | 12,790 | 123.8909 s | 0.744533 |
| Specialist only | 92% | 92% | 11,156 | 135.9041 s | 0.904118 |
| Learned router | 92% | 92% | 11,850 | 161.0610 s | 0.903084 |
| Two-model oracle | — | 96% | — | — | — |

The learned router used the frozen policy with SHA-256

```text
01428874602c7f5f02dd8cd59300b5d774b3452f0d225aa806a2f9591af9bb65
```

## Interpretation

The learned router improved strict correctness by 16 percentage points over the general Executor.

It matched the coding Specialist at 92% strict correctness.

It did not improve cost over the Specialist in the final test run. Specialist-only used fewer tokens, lower latency, and slightly higher average policy reward.

The two-model oracle reached 96%, showing that some complementarity remained available beyond the learned router.

## Development reference

On clean dev

| Method | Strict EvalPlus+ |
| --- | ---: |
| Executor only | 83.33% |
| Specialist only | 91.67% |
| Learned router | 95.83% |
| Two-model oracle | 95.83% |

The live end-to-end learned-router dev run reproduced the cached 95.83% result.

## Training reference

On the 90-task clean training split

| Method | Strict EvalPlus+ |
| --- | ---: |
| Executor only | 78.89% |
| Specialist only | 80.00% |
| Two-model oracle | 88.89% |

These training trajectories were used to fit the contextual router.

## Reporting rule

The final test results are accepted as observed.

The project does not tune prompts, models, reward weights, task features, tree hyperparameters, or split definitions after seeing the final test outcomes.
