# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/lilygeekgal/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/lilygeekgal/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two APIs, ten free providers: the best free model behind every request. It serves the OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in streaming, by routing each request to the highest-rated free model that meets its requirements, spreading calls over providers so no single free-tier budget is spent, and failing over to the next-best model when a provider refuses. Requests and answers are translated between the OpenAI, Anthropic and Gemini wires without dropping a field the provider echoes back. On the three practice trials it scores 99 (quality 99, served 100, wire 100, failover 100, catalog 100), with a byte-for-byte deterministic replay.

## Run it

```sh
node main.ts --list
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env> node main.ts
```

`main.ts` is the whole server: Node 22 standard library only, nothing to install. It reads the brief at `MAKEFUN_PROMPT_FILE`, probes each provider's catalog route, prints or registers the one catalog, serves the client's requests, polls the env until the workload is done, and exits 0.

---

<sub>Entered on [make.fun](https://make.fun/?ref=jhbw4fsb&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
