import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { StrandsTestInfraStack, StrandsTestInfraStackProps } from '../lib/stacks/test-infra-stack';
import { DEPLOY_ENVIRONMENTS, DIFF_ENVIRONMENTS } from '../lib/constructs/github-ci-roles';

let originalEnv: NodeJS.ProcessEnv;

beforeAll(() => {
  originalEnv = { ...process.env };
  process.env.STRANDS_TEST_INFRA_PRIVATE_REPOS = 'repo-a,repo-b';
  process.env.STRANDS_TEST_INFRA_BUCKET_NAMES = 'test-bucket-*';
  process.env.STRANDS_TEST_INFRA_PERSISTENT_BUCKET_NAMES = 'test-persistent-bucket-*,test-session-bucket-*';
  process.env.STRANDS_TEST_INFRA_SECRET_NAMES = 'test-secret';
});

afterAll(() => {
  process.env = originalEnv;
});

function synth(props?: Partial<ConstructorParameters<typeof StrandsTestInfraStack>[2]>): Template {
  const app = new cdk.App();
  const stack = new StrandsTestInfraStack(app, 'TestStack', {
    env: { account: '123456789012', region: 'us-east-1' },
    ...props,
  });
  return Template.fromStack(stack);
}

/** Every `Principal.AWS` ARN across the template's role trust policies. */
function assumeRolePrincipalArns(template: Template): string[] {
  return Object.values(template.findResources('AWS::IAM::Role'))
    .flatMap((role: any) => role.Properties?.AssumeRolePolicyDocument?.Statement ?? [])
    .map((statement: any) => statement.Principal?.AWS)
    .filter((arn: unknown): arn is string => typeof arn === 'string');
}

// --- Knowledge Base ---

test('creates an S3 vectors index matching Titan v2 (1024 dims, cosine, float32)', () => {
  const template = synth();

  template.hasResourceProperties('AWS::S3Vectors::Index', {
    DataType: 'float32',
    Dimension: 1024,
    DistanceMetric: 'cosine',
    MetadataConfiguration: {
      NonFilterableMetadataKeys: ['AMAZON_BEDROCK_TEXT', 'AMAZON_BEDROCK_METADATA'],
    },
  });
  template.resourceCountIs('AWS::S3Vectors::VectorBucket', 1);
});

test('knowledge base uses the S3 vectors store and Titan v2 embeddings', () => {
  const template = synth();

  template.hasResourceProperties('AWS::Bedrock::KnowledgeBase', {
    KnowledgeBaseConfiguration: {
      Type: 'VECTOR',
      VectorKnowledgeBaseConfiguration: {
        // ARN resolves to an Fn::Join whose final literal carries the model id.
        EmbeddingModelArn: {
          'Fn::Join': ['', Match.arrayWith([Match.stringLikeRegexp('amazon.titan-embed-text-v2:0$')])],
        },
        EmbeddingModelConfiguration: {
          BedrockEmbeddingModelConfiguration: { Dimensions: 1024, EmbeddingDataType: 'FLOAT32' },
        },
      },
    },
    StorageConfiguration: { Type: 'S3_VECTORS' },
  });
});

test('S3 data source points at the source bucket and retains data on deletion', () => {
  const template = synth();

  template.hasResourceProperties('AWS::Bedrock::DataSource', {
    Name: 'S3DataSource',
    DataDeletionPolicy: 'DELETE',
    DataSourceConfiguration: { Type: 'S3' },
  });
});

test('CUSTOM data source exists with no connection config', () => {
  const template = synth();

  template.hasResourceProperties('AWS::Bedrock::DataSource', {
    Name: 'CustomDataSource',
    DataDeletionPolicy: 'DELETE',
    DataSourceConfiguration: { Type: 'CUSTOM' },
  });
});

test('knowledge base service role can be assumed only by Bedrock in this account', () => {
  const template = synth();

  template.hasResourceProperties('AWS::IAM::Role', {
    AssumeRolePolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Principal: { Service: 'bedrock.amazonaws.com' },
          Condition: {
            StringEquals: { 'aws:SourceAccount': '123456789012' },
          },
        }),
      ]),
    },
  });
});

// --- SSH EC2 ---

test('SSH instance is t4g.nano with no public IP association', () => {
  const template = synth({ testFeatures: ['ssh-ec2'] });

  const instances = template.findResources('AWS::EC2::Instance');
  const props = (Object.values(instances)[0] as any).Properties;
  expect(props.InstanceType).toBe('t4g.nano');
  expect(props).not.toHaveProperty('NetworkInterfaces');
});

test('SSH VPC has three SSM interface endpoints and no NAT gateway', () => {
  const template = synth({ testFeatures: ['ssh-ec2'] });

  template.resourceCountIs('AWS::EC2::VPCEndpoint', 3);
  template.resourceCountIs('AWS::EC2::NatGateway', 0);
});

test('SSH grants StartSession, TerminateSession, OpenDataChannel, and private key read', () => {
  const template = synth({ internal: true, testFeatures: ['ssh-ec2'] });

  const policies = template.findResources('AWS::IAM::Policy');
  const statements = Object.values(policies).flatMap(
    (p: any) => p.Properties?.PolicyDocument?.Statement ?? [],
  );
  const actions = statements.flatMap((s: any) =>
    Array.isArray(s.Action) ? s.Action : [s.Action],
  );
  expect(actions).toEqual(expect.arrayContaining([
    'ssm:StartSession',
    'ssm:TerminateSession',
    'ssmmessages:OpenDataChannel',
    'ssm:GetParameter',
  ]));
});

test('SSH instance has no inbound security group rules', () => {
  const template = synth({ testFeatures: ['ssh-ec2'] });

  const ingress = template.findResources('AWS::EC2::SecurityGroupIngress');
  // The only SG ingress rules should be on the VPC endpoint SGs (self-referencing
  // for HTTPS), not custom rules we added for port 22.
  for (const rule of Object.values(ingress) as any[]) {
    expect(rule.Properties.FromPort).not.toBe(22);
  }
});

// --- Test Role ---

test('internal mode uses GitHub OIDC trust', () => {
  const template = synth({ internal: true, testFeatures: ['bedrock-knowledge-base'] });

  template.hasResourceProperties('AWS::IAM::Role', {
    AssumeRolePolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: 'sts:AssumeRoleWithWebIdentity',
          Principal: {
            Federated: Match.stringLikeRegexp('oidc-provider/token.actions.githubusercontent.com$'),
          },
        }),
      ]),
    },
  });
});

test('community mode uses AccountPrincipal trust', () => {
  const template = synth({ internal: false, testFeatures: ['bedrock-knowledge-base'] });

  template.hasResourceProperties('AWS::IAM::Role', {
    AssumeRolePolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Principal: { AWS: Match.objectLike({}) },
        }),
      ]),
    },
  });
});

test('internal mode with RUNNER_ROLES adds AssumeRole trust for those roles', () => {
  process.env.STRANDS_TEST_INFRA_RUNNER_ROLES = 'MyTestRunner';
  try {
    const template = synth({ internal: true, testFeatures: ['bedrock-knowledge-base'] });

    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'sts:AssumeRole',
            Principal: {
              AWS: Match.stringLikeRegexp(':role/MyTestRunner$'),
            },
          }),
        ]),
      },
    });
  } finally {
    delete process.env.STRANDS_TEST_INFRA_RUNNER_ROLES;
  }
});

test('internal mode refuses to synth when a required list is blank', () => {
  // A whitespace-only secret must not deploy a role with that list emptied.
  process.env.STRANDS_TEST_INFRA_SECRET_NAMES = '  ';
  try {
    expect(() => synth({ internal: true, testFeatures: ['bedrock-knowledge-base'] })).toThrow(
      /STRANDS_TEST_INFRA_SECRET_NAMES must be set/,
    );
  } finally {
    process.env.STRANDS_TEST_INFRA_SECRET_NAMES = 'test-secret';
  }
});

test('a blank RUNNER_ROLES value adds no trust principal', () => {
  // GitHub Actions passes an unset secret to a step as an empty string, which
  // must not become a principal with an empty role name.
  process.env.STRANDS_TEST_INFRA_RUNNER_ROLES = '';
  try {
    const template = synth({ internal: true, testFeatures: ['bedrock-knowledge-base'] });

    expect(assumeRolePrincipalArns(template).filter((arn) => arn.endsWith(':role/'))).toEqual([]);
  } finally {
    delete process.env.STRANDS_TEST_INFRA_RUNNER_ROLES;
  }
});

test('RUNNER_ROLES entries are trimmed and blanks between them ignored', () => {
  process.env.STRANDS_TEST_INFRA_RUNNER_ROLES = ' RunnerOne , ,RunnerTwo';
  try {
    const template = synth({ internal: true, testFeatures: ['bedrock-knowledge-base'] });

    expect(assumeRolePrincipalArns(template).filter((arn) => arn.includes(':role/Runner'))).toEqual([
      'arn:aws:iam::123456789012:role/RunnerOne',
      'arn:aws:iam::123456789012:role/RunnerTwo',
    ]);
  } finally {
    delete process.env.STRANDS_TEST_INFRA_RUNNER_ROLES;
  }
});

test('internal mode grants the Mantle actions the base-path drift tests need', () => {
  const template = synth({ internal: true });

  const policies = template.findResources('AWS::IAM::Policy');
  const actions = Object.values(policies).flatMap((p: any) =>
    (p.Properties?.PolicyDocument?.Statement ?? []).flatMap((s: any) => s.Action ?? []),
  );

  // The Mantle routing drift tests skip without ListModels. See #3654.
  expect(actions).toEqual(
    expect.arrayContaining([
      'bedrock-mantle:CreateInference',
      'bedrock-mantle:CallWithBearerToken',
      'bedrock-mantle:ListModels',
    ]),
  );
});

test('community mode does not attach the legacy broad policy', () => {
  const template = synth({ internal: false });

  const policies = template.findResources('AWS::IAM::Policy');
  for (const policy of Object.values(policies) as any[]) {
    const statements = policy.Properties?.PolicyDocument?.Statement ?? [];
    for (const stmt of statements) {
      expect(stmt.Action).not.toEqual(expect.arrayContaining(['aoss:CreateSecurityPolicy']));
    }
  }
});

test('persistent bucket policy grants access without DeleteBucket', () => {
  const template = synth({ internal: true });

  const policies = template.findResources('AWS::IAM::Policy');
  const statements = Object.values(policies).flatMap(
    (p: any) => p.Properties?.PolicyDocument?.Statement ?? [],
  );
  const persistentStmt = statements.find(
    (s: any) =>
      Array.isArray(s.Action) &&
      s.Action.includes('s3:PutObject') &&
      !s.Action.includes('s3:DeleteBucket') &&
      JSON.stringify(s.Resource).includes('test-persistent-bucket-*'),
  );
  expect(persistentStmt).toBeDefined();
  expect(persistentStmt.Action).toEqual(
    expect.arrayContaining(['s3:PutObject', 's3:GetObject', 's3:CreateBucket', 's3:ListBucket', 's3:DeleteObject']),
  );
  expect(persistentStmt.Action).not.toContain('s3:DeleteBucket');
});

// --- CI Roles ---

const SUB = 'token.actions.githubusercontent.com:sub';
const WORKFLOW_REF = 'token.actions.githubusercontent.com:job_workflow_ref';
const CI: Partial<StrandsTestInfraStackProps> = {
  internal: true,
  testFeatures: ['bedrock-knowledge-base'],
};

/** The StringEquals trust conditions of the named fixed-name role. */
function trust(template: Template, roleName: string): any {
  const role = Object.values(template.findResources('AWS::IAM::Role')).find(
    (r: any) => r.Properties?.RoleName === roleName,
  ) as any;
  expect(role).toBeDefined();
  const statements = role.Properties.AssumeRolePolicyDocument.Statement;
  expect(statements).toHaveLength(1);
  expect(statements[0].Action).toBe('sts:AssumeRoleWithWebIdentity');
  expect(statements[0].Principal.Federated).toMatch(
    /oidc-provider\/token\.actions\.githubusercontent\.com$/,
  );
  // StringEquals and nothing else: a stray StringLike is how an exact pin turns
  // into a wildcard escape hatch.
  expect(Object.keys(statements[0].Condition)).toEqual(['StringEquals']);
  return statements[0].Condition.StringEquals;
}

/** What the named role may assume, by bootstrap role name. */
function mayAssume(template: Template, logicalIdPrefix: string): string[] {
  const policies = Object.entries(template.findResources('AWS::IAM::Policy'))
    .filter(([id]) => id.startsWith(logicalIdPrefix))
    .map(([, p]) => p as any);
  expect(policies).toHaveLength(1);
  const statements = policies[0].Properties.PolicyDocument.Statement;
  expect(statements).toHaveLength(1);
  expect(statements[0].Action).toBe('sts:AssumeRole');
  return [statements[0].Resource].flat();
}

const WORKFLOW_REF_VALUE =
  'strands-agents/harness-sdk/.github/workflows/test-infra-deploy.yml@refs/heads/main';

// The subject is the environment, not the ref: pull_request_target reports the
// default branch in GITHUB_REF, so a ref subject cannot tell a reviewed push from
// a pull request. job_workflow_ref is what pins branch and file.
test('the deploy role trusts only the two deploy environments', () => {
  // Exhaustive: an unasserted extra condition key, or a missing `aud`, is how
  // this stops being an exact pin.
  expect(trust(synth(CI), 'StrandsTestInfraDeployRole')).toEqual({
    'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
    [SUB]: [
      'repo:strands-agents/harness-sdk:environment:test-infra-deploy',
      'repo:strands-agents/harness-sdk:environment:test-infra-deploy-approval',
    ],
    [WORKFLOW_REF]: WORKFLOW_REF_VALUE,
  });
});

// The diff job runs a pull request's own CDK code before anyone approves it, so
// its role is trusted for different environments — that disjointness is the only
// thing stopping a diff-job token from satisfying the deploy role's trust.
test('the diff role trusts only the authorization-check environments', () => {
  expect(trust(synth(CI), 'StrandsTestInfraDiffRole')).toEqual({
    'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
    [SUB]: [
      'repo:strands-agents/harness-sdk:environment:auto-approve',
      'repo:strands-agents/harness-sdk:environment:manual-approval',
    ],
    [WORKFLOW_REF]: WORKFLOW_REF_VALUE,
  });
});

// The subject must never pin the ref: pull_request_target reports the default
// branch in GITHUB_REF, so a ref subject cannot tell a reviewed push from a PR.
test('neither role trusts a bare ref subject', () => {
  const template = synth(CI);

  for (const roleName of ['StrandsTestInfraDeployRole', 'StrandsTestInfraDiffRole']) {
    expect(JSON.stringify(trust(template, roleName)[SUB])).not.toContain('ref:refs/heads');
  }
});

test('the diff and deploy environments are disjoint', () => {
  expect(DIFF_ENVIRONMENTS.filter((e) => DEPLOY_ENVIRONMENTS.includes(e))).toEqual([]);
});

test('the deploy role may assume the bootstrap roles, the diff role only lookup', () => {
  const template = synth(CI);
  const arn = (name: string) =>
    `arn:aws:iam::123456789012:role/cdk-hnb659fds-${name}-role-123456789012-us-east-1`;

  expect(mayAssume(template, 'StrandsTestInfraCiDeployRole')).toEqual(
    ['deploy', 'file-publishing', 'image-publishing', 'lookup'].map(arn),
  );
  // `deploy` can pass the CloudFormation execution role, so it stays out of reach
  // of pull-request code.
  expect(mayAssume(template, 'StrandsTestInfraCiDiffRole')).toEqual([arn('lookup')]);
});

test('community mode creates neither CI role', () => {
  const roleNames = Object.values(synth({ internal: false }).findResources('AWS::IAM::Role')).map(
    (role: any) => role.Properties?.RoleName,
  );

  expect(roleNames).not.toContain('StrandsTestInfraDeployRole');
  expect(roleNames).not.toContain('StrandsTestInfraDiffRole');
});

// --- Feature Toggling ---

test('selecting only bedrock-knowledge-base excludes EC2 resources', () => {
  const template = synth({ testFeatures: ['bedrock-knowledge-base'] });

  template.resourceCountIs('AWS::EC2::Instance', 0);
  template.resourceCountIs('AWS::Bedrock::KnowledgeBase', 1);
});

test('selecting only ssh-ec2 excludes KB resources', () => {
  const template = synth({ testFeatures: ['ssh-ec2'] });

  template.resourceCountIs('AWS::Bedrock::KnowledgeBase', 0);
  template.resourceCountIs('AWS::EC2::Instance', 1);
});

test('all (default) provisions both features', () => {
  const template = synth();

  template.resourceCountIs('AWS::Bedrock::KnowledgeBase', 1);
  template.resourceCountIs('AWS::EC2::Instance', 1);
});

// --- SSM Parameters ---

test('KB publishes ids and bucket name to SSM under its feature namespace', () => {
  const template = synth({ testFeatures: ['bedrock-knowledge-base'] });

  template.hasResourceProperties('AWS::SSM::Parameter', {
    Name: '/strands/test-infra/bedrock-knowledge-base/knowledge-base-id',
  });
  template.hasResourceProperties('AWS::SSM::Parameter', {
    Name: '/strands/test-infra/bedrock-knowledge-base/s3-data-source-id',
  });
  template.hasResourceProperties('AWS::SSM::Parameter', {
    Name: '/strands/test-infra/bedrock-knowledge-base/custom-data-source-id',
  });
  template.hasResourceProperties('AWS::SSM::Parameter', {
    Name: '/strands/test-infra/bedrock-knowledge-base/s3-source-bucket-name',
  });
});

test('SSH publishes instance-id and private-key-parameter-name to SSM under its feature namespace', () => {
  const template = synth({ testFeatures: ['ssh-ec2'] });

  template.hasResourceProperties('AWS::SSM::Parameter', {
    Name: '/strands/test-infra/ssh-ec2/instance-id',
  });
  template.hasResourceProperties('AWS::SSM::Parameter', {
    Name: '/strands/test-infra/ssh-ec2/private-key-parameter-name',
  });
});
