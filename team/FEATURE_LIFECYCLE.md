# Strands SDK Feature Lifecycle Process

## Overview

This document establishes a standardized process for adding new features and deprecating existing functionality in the Strands SDK while maintaining semantic versioning compliance and community trust.

## Vision for Major Releases

Major releases should be viewed as opportunities to reduce technical debt and remove deprecated features while maintaining a smooth upgrade path for the community. The goal is to enable users to upgrade from version 1.11 to 2.0 with minimal code changes and minimal behavioral differences.

Ideally, version 1.11 would include back-ported APIs from 2.0 to enable seamless migration.

### Support Implications

Major versions have significant implications for both maintainers and the community:

* **Maintainers**: Each major version comes with support requirements - major versions are supported for at least 6 months after the release of the next major version.
    * This includes bug fixes and security patches; new features are not back-ported to previous major versions
* **Community**: Major versions with breaking changes require effort from users to upgrade their applications

## Semantic Versioning Adherence

* **Major (X.0.0)**: Breaking changes, feature removals, API changes that affect existing code without user action
* **Minor (1.Y.0)**: New features, deprecation warnings, backward-compatible additions, and "pay for play" breaking changes (see Exceptions below)
* **Patch (1.1.Z)**: Bug fixes, security patches, documentation updates

## Exceptions to Versioning Adherence

### Rapidly Evolving Standards

While we strive for stability in the SDK, AI standards are evolving rapidly. To provide the community with the most relevant and capable features, we occasionally need to take dependencies on libraries or standards that are evolving and have features that may not adhere to strict versioning requirements. Examples include the OTEL GenAI Semantic Convention, MCP, and A2A protocols.

In these instances, we prioritize providing industry-standard implementations and adhering to specifications as closely as possible. If you're using features that depend on rapidly evolving standards, we recommend pinning to a specific minor version in production applications as a best practice.

### Pay for Play: Opt-In Breaking Changes

Small breaking changes that follow the "pay for play" principle are acceptable in minor versions without requiring a major version bump. This principle states: programs can call new APIs to access new features, but programs that choose not to do so are unaffected — old code continues to work as it did before.

**When This Applies:**

* The breaking change is gated behind new functionality that users must explicitly adopt
* Existing code paths remain completely unaffected
* Users who don't touch the new feature never observe the break
* The breakage is obvious and directly tied to something the user just added

**When This Does NOT Apply:**

* Existing code breaks without any user action (requires major version)
* The change affects default behavior (requires major version)
* Users upgrade and their code stops working with no obvious reason why (requires major version)

**Example**: Converting a `TypedDict` to `total=False` is technically breaking if existing implementations don't provide the new optional members. However, if the only way to encounter those new members is by adding a new tool that uses the new format, the change is effectively "pay for play." Users who don't adopt the new tool never observe the break.

**Rationale**: Strict semver adherence can slow SDK development when the breaking change only affects users who explicitly adopt new functionality. If existing code paths remain unaffected, the practical impact on the community is minimal, and the benefit of faster feature delivery outweighs the theoretical semver violation.

See also: [Raymond Chen on "pay for play" in API design](https://devblogs.microsoft.com/oldnewthing/20260127-00/?p=112018) 

## Feature Addition Process

### New Features

1. **Design Phase**: Create GitHub Discussion for significant features affecting public API to gather community input
2. **Implementation**: Develop with comprehensive tests and documentation
3. **Experimental Release**: Place new features in `strands.experimental` submodule
4. **Collect Feedback & Iterate**: Gather feedback from the community and iterate until the feature meets community needs and maintainer standards
5. **Monthly Review**: Regular cadence for evaluating experimental features
6. **Stabilization**: Move to main module in subsequent minor release after validation

### Feature Enhancement

* Backward-compatible improvements ship in minor versions
* For breaking changes to existing classes, evaluate adding new fields instead of modifying existing ones or introducing a new flag to enable new behavior
* Opt-in breaking changes that follow the "pay for play" principle (see Exceptions to Versioning Adherence) can ship in minor versions
* Breaking changes that affect existing code without user action require major version bump with migration guide

### What is experimental? 

The experimental module allows maintainers to move quickly, gather community feedback, and iterate on new features. We welcome community feedback and testing throughout this process.

Features marked as experimental are excluded from semantic versioning standards and backward compatibility guarantees between minor versions. While we generally recommend against using experimental features in production, if you choose to do so after proper vetting, ensure you pin to a specific minor version to avoid issues with breaking changes.

### How we use experimental

Experimental features allow maintainers to test new ideas, interfaces, and features with the community before formally including them in the SDK with full support. The goal of releasing an experimental feature is to gather community feedback on:

1. The utility of the feature
2. The shape of the feature (including interface and behavior)
3. The value it provides

When adding a feature to the experimental module, clear exit criteria must be defined for either promotion to the SDK or formal disqualification. Maintainers commit to proactively seeking community feedback for experimental features.

### Criteria for graduating from experimental

### Stability & Maturity

    - [ ] API is stable with no breaking changes expected
    - [ ] Core functionality is complete and tested

### Quality Assurance

    - [ ] Comprehensive test coverage (unit, integration, end-to-end)
    - [ ] Documentation is complete with examples and API references

### Usage & Adoption

    - [ ] At least 2-3 real-world use cases validated
    - [ ] Positive feedback from early adopters
    - [ ] No major unresolved issues or blockers
    - [ ] Clear value proposition demonstrated

### Technical Requirements

    - [ ] Error handling is robust and consistent
    - [ ] Logging and observability are implemented
    - [ ] Dependencies are stable and well-maintained

### Process & Governance

    - [ ] Migration path from experimental version documented
    - [ ] Deprecation timeline for experimental version established

### How do features graduate from experimental into the core SDK?

Once a feature has met the exit criteria and maintainers have determined it's ready for full SDK support:

* **Minor Version X.Y-1**: Feature only exists in the experimental module/status
* **Minor Version X.Y**: 
    * Experimental feature is copied to the primary SDK
    * The experimental version begins emitting deprecation warnings with `warnings.warn()` 
    * Documentation for the new feature is marked non-experimental with a migration guide
* **Minor Version X.Y+1**: Experimental feature is removed from the experimental module

## Deprecation Process

Releases within minor versions are additive and maintain backward compatibility. However, there are times when we need to remove features or rework functionality. The deprecation process ensures the community has adequate time to migrate:

* First, introduce an alternative/improved way to accomplish the feature
* Deprecate the old way with clear warnings and migration guidance
* Upon release of a new major version, remove the deprecated functionality

### Timeline

* **Major Version X:** Old way of doing something and new way of doing something introduced
    * **Minor Version X.Y-1**: Old way of doing something
    * **Minor Version X.Y**: New way of doing something is added, deprecation warning is applied to old way to with `@warnings.**deprecated**(*msg*, ***, *category=DeprecationWarning*, *stacklevel=1*) `
    * **Minor Version X.Y+1**: (optional) Enhance warning with migration examples or appropriate link to our docs pointing to the workaround
* **Major Version X+1**: Remove older, deprecated feature entirely



### Implementation Standards


Python

```python
import warnings

@warnings.deprecated( 
        "deprecated_function() is deprecated and will be removed in v2.0.0. "
        "Use new_function() instead. See migration guide: https://strands-agents.com/migration/deprecated_function", 
        *, 
        category=DeprecationWarning, 
        stacklevel=2
    ) 
    
```

Typescript

```typescript
/**
 * @deprecated deprecated_function() is deprecated and will be removed in v2.0.0. Use new_function() instead. See migration guide: https://strands-agents.com/migration/deprecated_function , 
 */
 export const deprecated_function = () => {}
```



### Communication Requirements

* **Changelog**: Document all deprecations with migration paths
* **Documentation**: Update with clear migration examples
* **GitHub Issues**: Create tracking issues for major deprecations
* **GitHub Discussions**: Use for gathering community feedback on proposed changes

## Community Trust Principles

These guiding principles inform every decision we make, honoring them in spirit and practice whenever possible.

### Predictability

* Maintain adequate versions transitions between deprecation warning and removal
* Never remove features in minor/patch versions
* Provide automated migration tools, clear error, or automatic fallback to new behavior when feasible

### Transparency

* Clear deprecation timelines in all warnings (example: will be removed in v2.0.0)
* Comprehensive migration documentation
* Regular communication through release notes

## Release Cadence

* **Patch releases**: As needed for critical fixes
* **Minor releases**: Regular cadence for new features and deprecation warnings, possible breaking changes for experimental features
* **Major releases**: With advance notice for breaking changes

## Rollback Policy

If a minor release causes significant community impact:

1. **Immediate patch release** to revert problematic changes
    * Yanking packages is possible but generally discouraged; forward patches are preferred
    * In cases where a mitigation cannot be provided quickly, a revert followed by a forward patch is the preferred approach
    * Yanking should only be used in extreme situations (e.g., critical security issues)
2. **Post-mortem analysis** to understand root cause
3. **Revised testing and acceptance criteria** for subsequent releases
