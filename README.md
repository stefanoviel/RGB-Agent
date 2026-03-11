# Read-Grep-Bash Agent

An agent for [ARC-AGI-3](https://three.arcprize.org/) that completes all three preview games in 1,069 actions, the lowest publicly reported count.

For details on approach and findings, see our [blog post](https://blog.alexisfox.dev/arcagi3).

![Architecture](assets/architecture.png)

## Setup

Requires Python (3.12 recommended) and Docker.

```bash
git clone git@github.com:alexisfox7/RGB-Agent.git
cd RGB-Agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
cd docker/opencode-sandbox && bash build.sh   # build analyzer sandbox image
```

To enable local packaged RE-ARC games from `Tufalabs/re-arc-3`, install the optional extra:

```bash
pip install -e ".[rearc]"
```

Create a `.env` file:

```
ARC_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Usage

```bash
arcgym-swarm --suite all --max-actions 500
arcgym-swarm --game ls20,ft09
arcgym-swarm --game rearc:snake
arcgym-swarm --game arc:ls20,rearc:ls20
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--suite` | — | Predefined game suites (e.g. `ls20`, `vc33`, `ft09`, or `all`) |
| `--game` | — | Comma-separated game names or IDs. Use `arc:` or `rearc:` prefixes when a short name exists in both catalogs. |
| `--max-actions` | 500 | Max actions per game |
| `--analyzer-interval` | 10 | Actions per analyzer batch plan |
| `--analyzer-model` | `auto` | Analyzer model (see below). `auto` prefers the local vLLM server if available. |
| `--operation-mode` | `online` | `online` / `offline` / `normal` |

Examples:

- `arcgym-swarm --game arc:ls20` runs the hosted ARC Prize `ls20-cb3b57cc` game.
- `arcgym-swarm --game rearc:ls20` runs the local RE-ARC `ls20-0001` packaged game.
- `arcgym-swarm --game snake-0001` works without a prefix when the ID is unique to RE-ARC.

### Analyzer models

`auto` is the default. It prefers a local OpenAI-compatible vLLM server discovered from `OPENAI_BASE_URL` or `/sw/public/vllm_server_registry/qwen3-32b-public.env`, and falls back to Anthropic if no local endpoint is available.

Anthropic models can still be passed without a prefix. For other providers, use `provider/model`.

| Model | `--analyzer-model` value |
|-------|--------------------------|
| Local vLLM default | `auto` (default) |
| Force local vLLM | `local` |
| Claude Opus 4.6 | `claude-opus-4-6` |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| GPT 5.2 | `openai/gpt-5.2` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |

Any model available via OpenRouter can also be used with the `openrouter/` prefix (e.g. `openrouter/google/gemini-3.1-pro-preview`).

Set the matching API key in `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `OPENROUTER_API_KEY`).

Results are saved to `evaluation_results/`.

### Qwen + OpenCode notes

`Qwen/Qwen3-32B` can be used as the analyzer through OpenCode, including OpenCode tool use (`read`, `grep`, `bash`), but the working setup is slightly different from the default Anthropic path.

What is required:

- A vLLM server with tool calling enabled.
- OpenCode available on the host (`~/.opencode/bin/opencode`) or in Docker.
- The Qwen compatibility path in this repo, which starts a local proxy for the OpenAI-compatible endpoint.

#### Starting a shared vLLM server on the cluster

Use `/home/stefano/slurm_vllm_server/run_vllm_server.sh`. That is the expected entrypoint for this repo on the cluster: it submits a Slurm batch job, starts vLLM inside the Pyxis container, and publishes the endpoint into the shared registry.

Recommended launch:

```bash
sbatch --partition=llm --account=llm --qos=llm \
  --ntasks=1 --gpus=1 --cpus-per-gpu=16 --mem-per-gpu=96G \
  /home/stefano/slurm_vllm_server/run_vllm_server.sh \
  --model Qwen/Qwen3-32B \
  --gpus 1 \
  --server-name qwen3-32b-public \
  --registry-dir /sw/public/vllm_server_registry \
  --max-model-len 40960 \
  --vllm-args "--enable-auto-tool-choice --tool-call-parser qwen3_xml"
```

Important flags:

- `--ntasks=1`, `--cpus-per-gpu=16`, `--mem-per-gpu=96G`: expected cluster resource shape for these GPU jobs.
- `--server-name qwen3-32b-public`: other users and jobs discover the endpoint by this name.
- `--registry-dir /sw/public/vllm_server_registry`: use a shared readable registry path if the server should be public on the cluster.
- `--max-model-len 40960`: working context size for `Qwen/Qwen3-32B` in this setup.
- `--vllm-args "--enable-auto-tool-choice --tool-call-parser qwen3_xml"`: enables the tool-calling path used by OpenCode.

To point the current shell at the published endpoint:

```bash
source /home/stefano/slurm_vllm_server/use_vllm_endpoint.sh \
  qwen3-32b-public /sw/public/vllm_server_registry
```

That sets `OPENAI_BASE_URL` from the registry entry.

#### Running this repo against the shared Qwen server

Example command for the seeded local RE-ARC `identify_the_agent` game:

```bash
OPENAI_BASE_URL=http://dgx01:8316/v1 \
OPENAI_API_KEY=EMPTY \
VLLM_MODEL_ID='Qwen/Qwen3-32B' \
OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=16000 \
PYTHONPATH=/tmp/re-arc-3 \
/home/stefano/RGB-Agent/.venv/bin/arcgym-swarm \
  --game rearc:identify_the_agent \
  --analyzer-model local \
  --max-actions 100 \
  --seed 7
```

If the server was published to the shared registry, you can usually omit the explicit `OPENAI_BASE_URL` after sourcing `use_vllm_endpoint.sh`:

```bash
source /home/stefano/slurm_vllm_server/use_vllm_endpoint.sh \
  qwen3-32b-public /sw/public/vllm_server_registry

OPENAI_API_KEY=EMPTY \
VLLM_MODEL_ID='Qwen/Qwen3-32B' \
OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=16000 \
PYTHONPATH=/tmp/re-arc-3 \
/home/stefano/RGB-Agent/.venv/bin/arcgym-swarm \
  --game rearc:identify_the_agent \
  --analyzer-model local \
  --max-actions 100 \
  --seed 7
```

Notes:

- `--seed` selects a deterministic game variant when the environment supports seeding.
- `--max-actions` is a total action budget for the full rollout, not per attempt. If the game resets after `GAME_OVER`, the action count keeps increasing.
- `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=16000` is useful for longer resumed analyzer sessions with local Qwen.

#### Why this was needed

With `Qwen/Qwen3-32B`, the original OpenCode streaming path failed on streamed tool-call deltas. The failure mode was that OpenCode expected the first streamed tool-call chunk to already contain `function.name`, while Qwen/vLLM emitted tool calls incrementally. That caused errors like:

- `Expected 'function.name' to be a string`
- later, stream termination failures around `data: [DONE]`

#### What changed in OpenCode

To make OpenCode work with `Qwen/Qwen3-32B`, the local OpenCode install was patched so the OpenAI-compatible provider does not rely on streamed tool-call parsing for this model.

The relevant change was in the installed `@ai-sdk/openai-compatible` bundle used by OpenCode:

- `doStream()` detects `Qwen3-32B`
- it internally calls `doGenerate()`
- it then emits a synthetic stream from the completed response

That means OpenCode still sees a stream, but Qwen tool calls are parsed from a complete response instead of fragile partial tool-call deltas.

There was also an earlier patch in OpenCode's Copilot-compatible provider path, but the generic OpenAI-compatible provider patch is the one that mattered for the `cluster-vllm` configuration used here.

After patching, OpenCode was rebuilt from `/home/stefano/opencode` and the host binary at `~/.opencode/bin/opencode` was replaced with the rebuilt version.

#### What changed in this repo

This repo adds a compatibility layer around the local Qwen endpoint:

- `arcgym/utils/qwen_tool_proxy.py` normalizes Qwen/vLLM responses for OpenCode
- `arcgym/agents/rgb_agent.py` can synthesize a final `[ACTIONS]` block from Qwen's analyzer output when the tool-using response does not end with valid action JSON
- the local Qwen analyzer path can reuse OpenCode sessions and sets `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` to keep resumed sessions within context limits

#### Practical behavior

The result is:

- OpenCode tools are used normally
- Qwen is still the analyzer model making the decisions
- this repo executes the resulting action plan

This fixes the transport and tool-calling compatibility problems. It does not guarantee that Qwen will solve every game; remaining failures are strategy-quality issues rather than OpenCode/vLLM protocol issues.
