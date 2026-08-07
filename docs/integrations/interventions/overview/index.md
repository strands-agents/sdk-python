An **intervention handler** is a composable control layer that intercepts agent lifecycle events and returns typed decisions — proceed, deny, guide, confirm, or transform. Handlers are registered via the `interventions` option in agent configuration and evaluated in order, with short-circuiting on deny or guide actions. See [Interventions](/docs/user-guide/concepts/agents/interventions/index.md) for the full concept and built-in handlers.

The SDK ships reference handlers like [Cedar Authorization](/docs/user-guide/concepts/agents/interventions/cedar-authorization/index.md), [Steering](/docs/user-guide/concepts/agents/interventions/steering/index.md), and [Human-in-the-Loop](/docs/user-guide/concepts/agents/interventions/human-in-the-loop/index.md). The packages below are **community-built** intervention handlers you can install and attach to an agent.

Community maintained

These packages are maintained by their authors, not the Strands team. Review packages before using them in production. Quality and support may vary.

## Browse the integrations page

See the [Interventions section of the integrations page](/integrations/?type=intervention/index.md) for the current list, with language support and links to each package.

## Add your intervention handler

Built an `InterventionHandler`? See the [Get Featured guide](/docs/integrations/get-featured/index.md) to list it here, and the [Extensions guide](/docs/contribute/contributing/extensions/index.md) for how to build and publish a package.