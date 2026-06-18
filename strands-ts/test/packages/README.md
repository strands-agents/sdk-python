# Package Import Tests

This directory contains verification tests to ensure `@strands-agents/sdk` can be imported correctly. There are two flavors, catching different classes of packaging bug:

- **`esm-module/` and `cjs-module/`** — fast local tests. Install the SDK via `file:../../..` and exercise ESM `import` + CommonJS `require`. Run by `npm run test:package`. These resolve through the monorepo, so they share the root `node_modules` and cannot detect missing-optional-peer regressions.
- **`npm-pack/`** — CI-only smoke test (`.github/workflows/test-package-pack.yml`). Runs `npm pack` and installs the tarball in a tempdir outside the monorepo, mirroring an end-user install. Catches the RC.0 class of bug where the main entry re-exports a symbol from an optional peer dependency.

## Running the Tests

From the root of the project:

```bash
npm run test:package
```

This command builds and installs the SDK locally, then runs both ESM and CJS import tests. The tarball test is not wired into this script — see `.github/workflows/test-package-pack.yml` for its invocation.

## Test Structure

```
test/packages/
├── esm-module/     # ES Module import test (file: install)
│   ├── esm.js      # Uses `import { ... } from '@strands-agents/sdk'`
│   └── package.json
├── cjs-module/     # CommonJS import test (file: install)
│   ├── cjs.js      # Uses `require('@strands-agents/sdk')`
│   └── package.json
├── npm-pack/       # Packed-tarball install smoke test (CI-only)
│   ├── verify.ts   # Type-checked consumer script
│   ├── package.json
│   └── tsconfig.json
└── README.md
```
