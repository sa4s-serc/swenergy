Agentic Efficiency Benchmarking

# **Why this problem?**

Agentic coding systems can now solve a substantial portion of SWE-bench Verified tasks. However, most evaluations still emphasize resolve rate while overlooking practical constraints such as latency, token usage, memory footprint, and energy consumption—all critical in real-world deployments. Frameworks like SWE-Agent and OpenHands have shown that interface and tooling design significantly shape both behavior and performance. Running these scaffolds under identical hardware and software conditions allows for meaningful comparisons that go beyond accuracy alone.

We propose a standardized evaluation of agentic code generation that explicitly accounts for energy efficiency. On a fixed workstation, we benchmark four open-source scaffolds paired with compatible open-weight models. For each SWE-bench Verified issue, we measure not only accuracy but also end-to-end energy usage, latency, token counts, and peak RAM/VRAM. This approach moves beyond accuracy-only reporting, providing reproducible, apples-to-apples comparisons across model–scaffold combinations.

# **Study design**
RQ1 — Energy-aware scaffold effectiveness. We first ask which of the three scaffolds—SWE-Agent, OpenHands, and AutoCodeRover—achieves the best ratio of task resolution to energy consumed when paired with the same 4B model. This focuses on resolution per joule, treating energy as a core evaluation dimension. By holding hardware and model fixed, we isolate differences that arise specifically from scaffold design. RQ2 — What drives scaffold efficiency? The second question investigates which scaffold-level factors explain variation in performance. In particular, we test whether differences in LLM/tool call counts and total tokens exchanged across scaffolds account for changes in wall-clock duration and total energy. We also consider peak RAM and VRAM usage as additional scaffold-induced contributors. This allows us to see how much of efficiency is determined by the scaffold’s orchestration rather than the underlying model. RQ3 — Scaffold trade-offs under resource constraints. Finally, we ask how the three scaffolds compare on the accuracy–efficiency frontier when jointly optimizing task resolution (↑) and total joules per issue (↓). We then examine how the frontier shifts once practical constraints such as maximum wall-clock duration or GPU memory limits are applied. This highlights which scaffold–model pairs remain competitive when deployed under realistic resource restrictions.
## **Hardware**

- Intel Xeon
- 32 GB RAM
- RTX A2000 16GB VRAM

## **Scaffolds**

- SWE-Agent
- OpenHands
- AutoCodeRover

## **Models**

- Gemma 3 12B IT (quantized to 8 bits)
- qwen3-30b-a3b
- ministral-8b-2410 (optional - if time permits)

## **Dataset**

- SWE-bench Verified mini(n=50)
- Single attempt per issue

## **Per-issue measurements**

- **Tokens:** total input/output (including tool/exec calls), logged by the scaffold or LLM server
- **Duration:** wall-clock time from first prompt to final pass/fail
- **CPU Time:** Time taken by the scaffold independent of the LLM
- **Energy:** GPU power via NVML, CPU via RAPL; reported in joules per issue
- **Memory:** peak VRAM (NVML) and peak RAM
- **Outcome**: pass/fail
- **Cost:** estimated via OpenRouter API rates × measured token counts
- **LLM Call Counts**
- **Tool Calls**

We have to run the entire experiment thrice.