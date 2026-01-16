// This file assumes that its run from an environment that already has github and core imported:
// const github = require('@actions/github');
// const core = require('@actions/core');

const fs = require('fs');

async function getIssueInfo(github, context, inputs) {
  const issueId = context.eventName === 'workflow_dispatch' 
    ? inputs.issue_id
    : context.payload.issue.number.toString();
  const commentBody = context.payload.comment?.body || '';
  const command = context.eventName === 'workflow_dispatch'
    ? inputs.command
    : (commentBody.startsWith('/strands') ? commentBody.slice('/strands'.length).trim() : '');

  console.log(`Event: ${context.eventName}, Issue ID: ${issueId}, Command: "${command}"`);

  const issue = await github.rest.issues.get({
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: issueId
  });

  return { issueId, command, issue };
}

async function determineBranch(github, context, issueId, mode, isPullRequest) {
  let branchName = 'main';

  if (mode === 'implementer' && !isPullRequest) {
    branchName = `agent-tasks/${issueId}`;
    
    const mainRef = await github.rest.git.getRef({
      owner: context.repo.owner,
      repo: context.repo.repo,
      ref: 'heads/main'
    });
    
    try {
      await github.rest.git.createRef({
        owner: context.repo.owner,
        repo: context.repo.repo,
        ref: `refs/heads/${branchName}`,
        sha: mainRef.data.object.sha
      });
      console.log(`Created branch ${branchName}`);
    } catch (error) {
      if (error.status === 422 || error.message?.includes('already exists')) {
        console.log(`Branch ${branchName} already exists`);
      } else {
        throw error;
      }
    }
  } else if (isPullRequest) {
    const pr = await github.rest.pulls.get({
      owner: context.repo.owner,
      repo: context.repo.repo,
      pull_number: issueId
    });
    branchName = pr.data.head.ref;
  }

  return branchName;
}

function buildPrompts(mode, issueId, isPullRequest, command, branchName, inputs) {
  const sessionId = inputs.session_id || (mode === 'implementer' 
    ? `${mode}-${branchName}`.replace(/[\/\\]/g, '-')
    : `${mode}-${issueId}`);

  const scriptFiles = {
    'implementer': '.github/agent-sops/task-implementer.sop.md',
    'refiner': '.github/agent-sops/task-refiner.sop.md',
    'release-notes': '.github/agent-sops/task-release-notes.sop.md'
  };
  
  const scriptFile = scriptFiles[mode] || scriptFiles['refiner'];
  const systemPrompt = fs.readFileSync(scriptFile, 'utf8');
  
  // Extract the user's feedback/instructions after the mode keyword
  // e.g., "release-notes Move #123 to Major Features" -> "Move #123 to Major Features"
  const modeKeywords = {
    'release-notes': /^(?:release-notes|release notes)\s*/i,
    'implementer': /^implement\s*/i,
    'refiner': /^refine\s*/i
  };
  
  const modePattern = modeKeywords[mode];
  const userFeedback = modePattern ? command.replace(modePattern, '').trim() : command.trim();
  
  let prompt = (isPullRequest) 
    ? 'The pull request id is:'
    : 'The issue id is:';
  prompt += `${issueId}\n`;
  
  // If there's any user feedback beyond the command keyword, include it as the main instruction, 
  // otherwise default to "review and continue"
  prompt += userFeedback || 'review and continue';

  return { sessionId, systemPrompt, prompt };
}

module.exports = async (context, github, core, inputs) => {
  try {
    const { issueId, command, issue } = await getIssueInfo(github, context, inputs);
    
    const isPullRequest = !!issue.data.pull_request;
    
    // Determine mode based on explicit command first, then context
    let mode;
    if (command.startsWith('release-notes') || command.startsWith('release notes')) {
      mode = 'release-notes';
    } else if (command.startsWith('implement')) {
      mode = 'implementer';
    } else if (command.startsWith('refine')) {
      mode = 'refiner';
    } else {
      // Default behavior when no explicit command: PR -> implementer, Issue -> refiner
      mode = isPullRequest ? 'implementer' : 'refiner';
    }
    console.log(`Is PR: ${isPullRequest}, Command: "${command}", Mode: ${mode}`);

    const branchName = await determineBranch(github, context, issueId, mode, isPullRequest);
    console.log(`Building prompts - mode: ${mode}, issue: ${issueId}, is PR: ${isPullRequest}`);

    const { sessionId, systemPrompt, prompt } = buildPrompts(mode, issueId, isPullRequest, command, branchName, inputs);
    
    console.log(`Session ID: ${sessionId}`);
    console.log(`Task prompt: "${prompt}"`);

    core.setOutput('branch_name', branchName);
    core.setOutput('session_id', sessionId);
    core.setOutput('system_prompt', systemPrompt);
    core.setOutput('prompt', prompt);

  } catch (error) {
    const errorMsg = `Failed: ${error.message}`;
    console.error(errorMsg);
    core.setFailed(errorMsg);
  }
};
