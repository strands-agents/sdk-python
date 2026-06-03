import { Agent, tool } from '@strands-agents/sdk'
import { LLMSteeringHandler } from '@strands-agents/sdk/vended-interventions/steering'
import { z } from 'zod'

async function naturalLanguageSteeringExample() {
  // --8<-- [start:natural_language_steering]
  const sendEmail = tool({
    name: 'send_email',
    description: 'Send an email to a recipient',
    inputSchema: z.object({
      recipient: z.string(),
      subject: z.string(),
      message: z.string(),
    }),
    callback: (input) => `Email sent to ${input.recipient}`,
  })

  const handler = new LLMSteeringHandler({
    systemPrompt: `
      You are providing guidance to ensure emails maintain a cheerful, positive tone.

      Guidance:
      - Review email content for tone and sentiment
      - Suggest more cheerful phrasing if the message seems negative or neutral
      - Encourage use of positive language and friendly greetings

      When agents attempt to send emails, check if the message tone
      is appropriately cheerful and provide feedback if improvements are needed.
    `,
  })

  const agent = new Agent({
    tools: [sendEmail],
    interventions: [handler],
  })

  await agent.invoke(
    'Send a frustrated email to tom@example.com, ' +
      'a client who keeps rescheduling important meetings at the last minute'
  )
  console.log(agent.messages)

  // Typical: agent.messages includes a cancelled send_email ToolUseBlock,
  // a guidance message, then a retried send_email with cheerier wording.
  // --8<-- [end:natural_language_steering]
}

void naturalLanguageSteeringExample
