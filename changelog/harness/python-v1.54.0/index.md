# Harness Python v1.54.0

Released 2026-08-27
Release: https://github.com/strands-agents/harness-sdk/releases/tag/python/v1.54.0 · Package: https://pypi.org/project/strands-agents/1.54.0/

## Features
- add cache\_key to CacheConfig for key-routed cache providers [model] (https://github.com/strands-agents/harness-sdk/pull/3949)
- add configurable classifier strategy [multiagent, model] (https://github.com/strands-agents/harness-sdk/pull/3846)
- add FileMemoryStore [persistence] (https://github.com/strands-agents/harness-sdk/pull/3925)
- inject an external cancellation signal into agent invocations [async, agent] (https://github.com/strands-agents/harness-sdk/pull/3999)
- add session\_id property to Agent [agent, sessions] (https://github.com/strands-agents/harness-sdk/pull/4007)

## Fixes
- count Gemini tool-use tokens as input and thinking tokens as output [model] (https://github.com/strands-agents/harness-sdk/pull/3892)
- warn and skip cachePoint message blocks instead of raising [model] (https://github.com/strands-agents/harness-sdk/pull/3947)
- redact blocked content when guardrail trace is disabled [model, interventions] (https://github.com/strands-agents/harness-sdk/pull/3772)
- convert JSON content blocks to text for Amazon Nova models in tool results [model] (https://github.com/strands-agents/harness-sdk/pull/1985)
- harden mantle routing integ tests (https://github.com/strands-agents/harness-sdk/pull/3957)
- restore always() in upload-metrics if guard (https://github.com/strands-agents/harness-sdk/pull/3970)
- yield trailing usage metadata chunk after finish\_reason (#3902) [model] (https://github.com/strands-agents/harness-sdk/pull/3955)
- grant Bidi integration test permissions [bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3975)
- use shared Google API key in integration test [model, bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3977)
- append the format to message document filenames [model] (https://github.com/strands-agents/harness-sdk/pull/3912)
- prevent in-place mutation of message history in normalizer [context, agent] (https://github.com/strands-agents/harness-sdk/pull/2326)
- detect Mantle context overflow errors [context, model] (https://github.com/strands-agents/harness-sdk/pull/3973)
- emit semconv-compliant cache usage attributes [otel] (https://github.com/strands-agents/harness-sdk/pull/3964)
- simplify agent delegation by making end\_turn accept content blocks [multiagent, hooks] (https://github.com/strands-agents/harness-sdk/pull/3961)
- treat empty integration test reports as a no-op (https://github.com/strands-agents/harness-sdk/pull/4006)
- count cached tokens in context-size and compaction baseline [context] (https://github.com/strands-agents/harness-sdk/pull/3886)

## Other
- refresh dependencies and isolate test results (https://github.com/strands-agents/harness-sdk/pull/3900)
- bump dorny/paths-filter from 3.0.2 to 4.0.3 (https://github.com/strands-agents/harness-sdk/pull/3907)
- harden bedrock kb integ tests [persistence] (https://github.com/strands-agents/harness-sdk/pull/3954)
- consolidate development configuration [bidirectional-streaming] (https://github.com/strands-agents/harness-sdk/pull/3960)
- update Bidi Bedrock dependencies (https://github.com/strands-agents/harness-sdk/pull/3997)
- record decision on null vs undefined input handling (https://github.com/strands-agents/harness-sdk/pull/3889)
