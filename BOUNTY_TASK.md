# Bounty Objective: config: kto and reward_model still accept llava / sharegpt4v rows and fail after the model loads
Target: MakazhanAlpamys/Soup Issue #2078
URL: https://github.com/MakazhanAlpamys/Soup/issues/2078

## Issue Description
### Summary

Follow-up to #1867 / #2064. #2064 refuses the vision row formats `llava` and `sharegpt4v` at config load for `dpo`, `ipo`, `simpo`, `orpo`, `bco` and `preference`, because Soup has no vision-preference row path. Two more tasks take the same row shape and are not covered: `kto` and `reward_model`. With `data.format: llava` or `sharegpt4v`, both still load and then fail late, after the model has loaded, because neither trainer has any image handling.

### Expected

The same refusal at config load for `kto` and `reward_model`, using the same validator and message shape as #2064 (the message names the task and the format, and suggests a text format). Before extending it, check `data.format: auto` and the `video` / `multimodal` formats too, and either cover them or say in the pull request why they cannot reach these tasks.

### Acceptance

- Refusal tests for `kto` and `reward_model` with `llava` and `sharegpt4v`, asserting `task=...` and `data.format=...` in the message, plus text-format controls that still load.
- Mutation check: drop either task from the refused set, and its test must fail.
- A changelog fragment, since configs that loaded before are now refused at load (they could not train).
- No recipe changes are expected; confirm that no recipe in the catalog pairs these tasks with a vision row format.

`area:training`: @ybai08 has the first refusal (3 days). After that, it is open to a claim comment with a short plan; @Vishalkumarsingh6428 offered on #2064.


## Instructions for Kuro (Autonomous Solver):
1. Use skills: `systematic-debugging` and `code-review-and-quality`.
2. Inspect the repository structure, locate relevant files, and understand the bug/feature.
3. Apply the minimal, robust fix or implementation.
4. Run any existing test suites or linters in the workspace.
5. Ensure the fix adheres to the project's existing code style.
6. When done, output a summary of files modified and verification results.
