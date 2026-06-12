// Run with: node --test .github/scripts/bot-pr-review-gate.test.mjs
import test from 'node:test';
import assert from 'node:assert';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const botPrReviewGate = require('./bot-pr-review-gate.cjs');

// Build mocked github/context/core for a scenario.
function makeDeps({
  baseRef = 'main',
  author = 'alice',
  authorType = 'User',
  labels = [],
  performedViaApp = null,
  commitEmails = ['alice@example.com'],
  reviews = [], // [{login, state}]
  roles = {}, // login -> role_name
  permErrorFor = new Set(),
  existingComments = [],
} = {}) {
  let failure = null;
  const createdComments = [];
  const updatedComments = [];

  const context = {
    repo: { owner: 'strands-agents', repo: 'harness-sdk' },
    payload: {
      pull_request: {
        number: 123,
        base: { ref: baseRef, sha: 'basesha' },
        user: { login: author, type: authorType },
        labels: labels.map((name) => ({ name })),
        performed_via_github_app: performedViaApp,
      },
    },
  };

  // Tag list endpoints so paginate can dispatch on identity.
  const listCommits = () => {};
  const listReviews = () => {};
  const listComments = () => {};

  const github = {
    rest: {
      pulls: { listCommits, listReviews },
      issues: {
        listComments,
        createComment: async (args) => { createdComments.push(args); },
        updateComment: async (args) => { updatedComments.push(args); },
      },
      repos: {
        getCollaboratorPermissionLevel: async ({ username }) => {
          if (permErrorFor.has(username)) throw new Error('Not Found');
          return { data: { role_name: roles[username] ?? 'read' } };
        },
      },
    },
    paginate: async (fn) => {
      if (fn === listCommits) return commitEmails.map((email) => ({ commit: { author: { email } } }));
      if (fn === listReviews) return reviews.map((r) => ({ user: { login: r.login }, state: r.state }));
      if (fn === listComments) return existingComments;
      throw new Error('unexpected paginate target');
    },
  };

  const core = {
    info: () => {},
    setFailed: (msg) => { failure = msg; },
  };

  const env = { REQUIRED_APPROVALS: '2', MAINTAINER_ROLES: 'maintain,admin', BASE_BRANCH: 'main', POST_COMMENT: 'true' };

  return { github, context, core, env, getFailure: () => failure, createdComments, updatedComments };
}

async function run(opts) {
  const deps = makeDeps(opts);
  await botPrReviewGate(deps);
  return deps;
}

test('human author with no bot signals is a no-op (allowed)', async () => {
  const { getFailure, createdComments } = await run({ author: 'alice', authorType: 'User', commitEmails: ['alice@corp.com'] });
  assert.strictEqual(getFailure(), null);
  assert.strictEqual(createdComments.length, 0, 'should not comment on human PRs');
});

test('human author using GitHub private email is NOT flagged (regression for noreply false-positive)', async () => {
  const { getFailure } = await run({
    author: 'realhuman',
    authorType: 'User',
    commitEmails: ['12345+realhuman@users.noreply.github.com'],
  });
  assert.strictEqual(getFailure(), null);
});

test('human handle containing "agent"/"bot" substrings is NOT flagged', async () => {
  for (const name of ['Talbot', 'realestate-agent', 'automation-fan', 'abbot']) {
    const { getFailure } = await run({ author: name, authorType: 'User', commitEmails: ['x@corp.com'] });
    assert.strictEqual(getFailure(), null, `${name} should not be gated`);
  }
});

test('Bot user type with 0 approvals is blocked and gets a comment', async () => {
  const { getFailure, createdComments } = await run({ author: 'some-app[bot]', authorType: 'Bot', reviews: [] });
  assert.match(getFailure(), /2 maintainer approvals/);
  assert.strictEqual(createdComments.length, 1);
  assert.match(createdComments[0].body, /0\/2/);
});

test('dependabot[bot] with 2 maintainer approvals is allowed', async () => {
  const { getFailure } = await run({
    author: 'dependabot[bot]',
    authorType: 'Bot',
    reviews: [
      { login: 'm1', state: 'APPROVED' },
      { login: 'm2', state: 'APPROVED' },
    ],
    roles: { m1: 'maintain', m2: 'admin' },
  });
  assert.strictEqual(getFailure(), null);
});

test('agent PR via label, 1 approval, is blocked', async () => {
  const { getFailure } = await run({
    author: 'realhuman',
    authorType: 'User',
    labels: ['agent'],
    reviews: [{ login: 'm1', state: 'APPROVED' }],
    roles: { m1: 'maintain' },
  });
  assert.match(getFailure(), /1\/2/);
});

test('approvals from non-maintainers do not count', async () => {
  const { getFailure } = await run({
    author: 'bot[bot]',
    authorType: 'Bot',
    reviews: [
      { login: 'c1', state: 'APPROVED' },
      { login: 'c2', state: 'APPROVED' },
    ],
    roles: { c1: 'write', c2: 'read' },
  });
  assert.match(getFailure(), /0\/2/);
});

test('COMMENTED review does NOT revoke a prior APPROVED (GitHub semantics)', async () => {
  const { getFailure } = await run({
    author: 'thing[bot]',
    authorType: 'Bot',
    reviews: [
      { login: 'm1', state: 'APPROVED' },
      { login: 'm2', state: 'APPROVED' },
      { login: 'm2', state: 'COMMENTED' }, // later comment must not drop approval
    ],
    roles: { m1: 'maintain', m2: 'admin' },
  });
  assert.strictEqual(getFailure(), null);
});

test('CHANGES_REQUESTED after APPROVED does revoke the approval', async () => {
  const { getFailure } = await run({
    author: 'thing[bot]',
    authorType: 'Bot',
    reviews: [
      { login: 'm1', state: 'APPROVED' },
      { login: 'm2', state: 'APPROVED' },
      { login: 'm2', state: 'CHANGES_REQUESTED' },
    ],
    roles: { m1: 'maintain', m2: 'admin' },
  });
  assert.match(getFailure(), /1\/2/);
});

test('bot commit email beyond first page is detected (pagination)', async () => {
  const emails = Array.from({ length: 30 }, (_, i) => `human${i}@corp.com`);
  emails.push('svc[bot]@users.noreply.github.com'); // 31st commit
  const { getFailure } = await run({ author: 'realhuman', authorType: 'User', commitEmails: emails });
  assert.match(getFailure(), /2 maintainer approvals/);
});

test('non-main base branch is skipped', async () => {
  const { getFailure } = await run({ baseRef: 'release/1.x', author: 'some[bot]', authorType: 'Bot' });
  assert.strictEqual(getFailure(), null);
});

test('permission API error treats reviewer as non-maintainer', async () => {
  const { getFailure } = await run({
    author: 'auto[bot]',
    authorType: 'Bot',
    reviews: [
      { login: 'm1', state: 'APPROVED' },
      { login: 'ghost', state: 'APPROVED' },
    ],
    roles: { m1: 'maintain' },
    permErrorFor: new Set(['ghost']),
  });
  assert.match(getFailure(), /1\/2/);
});

test('status comment is reused (updated, not duplicated) when one already exists', async () => {
  const { createdComments, updatedComments } = await run({
    author: 'some[bot]',
    authorType: 'Bot',
    reviews: [],
    existingComments: [{ id: 99, body: `${botPrReviewGate.COMMENT_MARKER}\nold body` }],
  });
  assert.strictEqual(createdComments.length, 0);
  assert.strictEqual(updatedComments.length, 1);
  assert.strictEqual(updatedComments[0].comment_id, 99);
});
