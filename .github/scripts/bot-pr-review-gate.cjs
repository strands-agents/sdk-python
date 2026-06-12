'use strict';

/**
 * Bot/Agent PR review gate.
 *
 * Fails the check unless a PR authored by a bot or agent has received the
 * required number of maintainer approvals. Human-authored PRs are a no-op.
 *
 * Maintainers are resolved via repos.getCollaboratorPermissionLevel
 * (role_name), mirroring the repo's authorization-check action, so the default
 * GITHUB_TOKEN is sufficient (no org-team read scope required).
 *
 * Detection deliberately favors structured, low-false-positive signals
 * (GitHub user type, the reserved `[bot]` suffix, GitHub App origin, and
 * explicit automation labels) over fuzzy username/email substring matching.
 *
 * @param {object} deps
 * @param {object} deps.github - Octokit instance (actions/github-script `github`).
 * @param {object} deps.context - Workflow context (actions/github-script `context`).
 * @param {object} deps.core - Actions core toolkit (actions/github-script `core`).
 * @param {object} [deps.env] - Environment overrides (defaults to process.env).
 */
async function botPrReviewGate({ github, context, core, env = process.env }) {
  const pr = context.payload.pull_request;
  if (!pr) {
    core.info('No pull_request in payload — nothing to do.');
    return;
  }

  const requiredApprovals = parseInt(env.REQUIRED_APPROVALS || '2', 10);
  const maintainerRoles = (env.MAINTAINER_ROLES || 'maintain,admin')
    .split(',')
    .map((r) => r.trim())
    .filter(Boolean);
  const baseBranch = env.BASE_BRANCH || 'main';
  const postComment = (env.POST_COMMENT || 'true') === 'true';

  if (pr.base.ref !== baseBranch) {
    core.info(`PR targets ${pr.base.ref}, not ${baseBranch} — skipping.`);
    return;
  }

  const author = pr.user.login;
  const authorType = pr.user.type;

  // Anchored, named bot accounts + the `[bot]` suffix GitHub reserves for
  // GitHub App accounts. Generic substrings (bot$, agent, automation) were
  // intentionally removed: they match human handles (Talbot, *-agent, ...).
  const botUsernamePatterns = [
    /\[bot\]$/, // GitHub reserves this suffix for app accounts
    /^dependabot(\[bot\])?$/i,
    /^renovate(\[bot\])?$/i,
    /^github-actions(\[bot\])?$/i,
    /^snyk-bot$/i,
    /^codecov(\[bot\])?$/i,
    /^mergify(\[bot\])?$/i,
    /^greenkeeper(\[bot\])?$/i,
  ];

  // Only the reserved `[bot]@` email shape. The broad noreply patterns were
  // removed because `…@users.noreply.github.com` is the default privacy email
  // for *human* accounts, which would gate ordinary contributors.
  const botEmailPatterns = [/\[bot\]@/i];

  // Labels a maintainer/automation can apply to opt a PR into the gate.
  const botLabels = new Set(['bot', 'dependencies', 'automated', 'auto-generated', 'agent']);

  const reasons = [];

  if (authorType === 'Bot') {
    reasons.push('user type is "Bot"');
  }

  if (botUsernamePatterns.some((p) => p.test(author))) {
    reasons.push(`username "${author}" matches a known bot account`);
  }

  const prLabels = (pr.labels || []).map((l) => l.name.toLowerCase());
  const matchedLabels = prLabels.filter((l) => botLabels.has(l));
  if (matchedLabels.length > 0) {
    reasons.push(`PR has label(s): ${matchedLabels.join(', ')}`);
  }

  if (pr.performed_via_github_app) {
    reasons.push(`PR created via GitHub App "${pr.performed_via_github_app.name}"`);
  }

  // Inspect all commit author emails (paginated — not just the first page).
  const commits = await github.paginate(github.rest.pulls.listCommits, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: pr.number,
  });
  const botEmails = commits
    .map((c) => c.commit.author && c.commit.author.email)
    .filter((email) => email && botEmailPatterns.some((p) => p.test(email)));
  if (botEmails.length > 0) {
    const unique = [...new Set(botEmails)];
    reasons.push(`commit author email(s) match bot pattern: ${unique.join(', ')}`);
  }

  const isBot = reasons.length > 0;
  core.info(`PR author: ${author}, type: ${authorType}, isBot: ${isBot}`);
  if (isBot) {
    core.info(`Detection reasons:\n  - ${reasons.join('\n  - ')}`);
  }

  if (!isBot) {
    core.info('PR is not from a bot/agent — no additional review requirement.');
    return;
  }

  // Count maintainer approvals. Only the latest review per user is considered.
  // COMMENTED reviews are ignored because, per GitHub's own merge semantics, a
  // follow-up comment does not revoke an existing APPROVED state.
  const reviews = await github.paginate(github.rest.pulls.listReviews, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: pr.number,
  });

  const latestReviewByUser = new Map();
  for (const review of reviews) {
    if (!review.user) continue;
    if (review.state === 'COMMENTED') continue;
    latestReviewByUser.set(review.user.login, review.state);
  }

  async function isMaintainer(username) {
    try {
      const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
        owner: context.repo.owner,
        repo: context.repo.repo,
        username,
      });
      return maintainerRoles.includes(data.role_name);
    } catch (error) {
      core.info(`Failed to check permission for ${username}: ${error.message}`);
      return false;
    }
  }

  const approvedBy = [];
  for (const [login, state] of latestReviewByUser.entries()) {
    if (state !== 'APPROVED') continue;
    if (await isMaintainer(login)) {
      approvedBy.push(login);
    }
  }

  const satisfied = approvedBy.length >= requiredApprovals;
  core.info(`Maintainer approvals: ${approvedBy.length}/${requiredApprovals}`);
  core.info(`Approved by: ${approvedBy.join(', ') || 'none'}`);

  if (postComment) {
    await upsertStatusComment({
      github,
      context,
      core,
      prNumber: pr.number,
      satisfied,
      approvedBy,
      requiredApprovals,
      maintainerRoles,
      reasons,
    });
  }

  if (!satisfied) {
    core.setFailed(
      `Bot/agent PRs require at least ${requiredApprovals} maintainer approvals before merging. ` +
        `Currently have ${approvedBy.length}/${requiredApprovals}. ` +
        `Maintainers are repository collaborators with role: ${maintainerRoles.join(' or ')}.`
    );
  }
}

const COMMENT_MARKER = '<!-- bot-pr-review-gate -->';

/**
 * Create or update a single sticky status comment so authors understand why
 * the gate is blocking and how many approvals remain. Idempotent: reuses the
 * existing marked comment instead of posting a new one on every run.
 */
async function upsertStatusComment({
  github,
  context,
  core,
  prNumber,
  satisfied,
  approvedBy,
  requiredApprovals,
  maintainerRoles,
  reasons,
}) {
  const body = satisfied
    ? `${COMMENT_MARKER}\n` +
      `✅ **Bot/agent PR review requirement satisfied** ` +
      `(${approvedBy.length}/${requiredApprovals} maintainer approvals).`
    : `${COMMENT_MARKER}\n` +
      `🤖 This PR was detected as bot/agent-authored, so it requires ` +
      `**${requiredApprovals} maintainer approvals** before merging ` +
      `(currently **${approvedBy.length}/${requiredApprovals}**).\n\n` +
      `Maintainers are repository collaborators with role: ${maintainerRoles.join(' or ')}.\n\n` +
      `<details><summary>Why was this flagged?</summary>\n\n- ${reasons.join('\n- ')}\n</details>`;

  try {
    const comments = await github.paginate(github.rest.issues.listComments, {
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: prNumber,
    });
    const existing = comments.find((c) => c.body && c.body.includes(COMMENT_MARKER));

    if (existing) {
      if (existing.body !== body) {
        await github.rest.issues.updateComment({
          owner: context.repo.owner,
          repo: context.repo.repo,
          comment_id: existing.id,
          body,
        });
      }
    } else {
      await github.rest.issues.createComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: prNumber,
        body,
      });
    }
  } catch (error) {
    // Commenting is best-effort; never let it mask the gate result.
    core.info(`Could not upsert status comment: ${error.message}`);
  }
}

module.exports = botPrReviewGate;
module.exports.COMMENT_MARKER = COMMENT_MARKER;
