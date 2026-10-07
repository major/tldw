# Merge queue ruleset (cutover config)

This file is the **versioned source of truth** for the GitHub merge queue on
`main`. It is **not applied automatically** — enabling the queue is a
deliberate, gated cutover because GitHub only allows the `merge_queue` rule
on **organization-owned public repos** (or Enterprise Cloud private repos).
`major/tldw` is currently user-owned, so the API rejects the rule today with
`422 {"errors":["Invalid rule 'merge_queue': "]}`. See
[GitHub: Managing a merge queue][mq-docs] for the availability rules.

[mq-docs]: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue

## Prerequisites

1. The repo is **transferred to a GitHub organization** (free for public
   repos). Until then, the rule will not be accepted by the API.
2. Repo setting `Allow auto-merge` is on (already `true` for `major/tldw`).
3. `.github/workflows/main.yml` has a `merge_group` trigger so CI runs on the
   speculative merge commit. That change is included in this same branch.

## Ruleset shape

| Rule | Why it is here |
|---|---|
| `deletion` | Block direct deletion of `main`. |
| `non_fast_forward` | Block force-pushes that would rewrite history. |
| `required_status_checks` (`Python quality`, `container`) | The queue validates these on every speculative merge commit. Mirrors the existing `CI` ruleset (id `24661567`). |
| `pull_request` (1 approval, dismiss stale, resolve threads) | The queue only acts on PRs, and merges should require a real human sign-off. |
| `merge_queue` | The queue itself. See parameter notes below. |

### Merge queue parameters

| Parameter | Value | Why |
|---|---|---|
| `merge_method` | `SQUASH` | Matches the repo's existing squash preference; the commit shape of `main` is unchanged. |
| `max_entries_to_build` | `5` | Caps speculative CI runs. The runner-safety knob — lower if runners get throttled. |
| `max_entries_to_merge` | `5` | Cap the size of a single merge group. |
| `min_entries_to_merge` | `1` | A lone PR is never blocked waiting for a batch to fill. |
| `min_entries_to_merge_wait_minutes` | `5` | Brief window to let a batch coalesce before merging a single entry. |
| `grouping_strategy` | `ALLGREEN` | A batch only merges if every PR in the group passes; GitHub bisects on failure. Use `HEADGREEN` if you only want the final combined commit validated (faster, riskier). |
| `check_response_timeout_minutes` | `60` | A required check that never reports fails the entry after 60 minutes instead of wedging the queue. |

## Apply

### Option A — replace the existing `CI` ruleset (recommended)

The repo already has a `CI` ruleset (id `24661567`) covering the same checks
and history protections. Replacing it in place keeps a single ruleset on
`main` and avoids a confusing two-ruleset state during cutover.

```bash
# Find the ruleset id (sanity check, expected 24661567)
gh api repos/<org>/tldw/rulesets --jq '.[] | {id, name, enforcement}'

# Apply
gh api -X PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  repos/<org>/tldw/rulesets/24661567 \
  --input .github/merge-queue-ruleset.json
```

### Option B — create a new ruleset alongside the existing one

Useful if you want to compare old vs new behaviour before retiring `CI`.

```bash
gh api -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  repos/<org>/tldw/rulesets \
  --input .github/merge-queue-ruleset.json
```

Then delete the old `CI` ruleset when ready:

```bash
gh api -X DELETE repos/<org>/tldw/rulesets/24661567
```

## Verify

```bash
# Confirm the merge_queue rule is now live
gh api repos/<org>/tldw/rulesets/24661567 \
  --jq '.rules[] | select(.type == "merge_queue")'

# Confirm CI has a merge_group trigger
gh api repos/<org>/tldw/contents/.github/workflows/main.yml \
  --jq '.content' | base64 -d | grep -A1 merge_group

# First smoke test: open a tiny PR and watch the queue
gh pr create --draft --title "smoke: merge queue" --body "queue check"
gh pr ready
```

## Pause / rollback

Operational pause (e.g. runners are down): flip `enforcement` to
`"disabled"`. The rules stay in place, nothing is forced.

```bash
gh api -X PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  repos/<org>/tldw/rulesets/24661567 \
  --input <(jq '.enforcement = "disabled"' .github/merge-queue-ruleset.json)
```

Full rollback: drop the `merge_queue` rule from the file and re-apply, or
delete the ruleset entirely.
