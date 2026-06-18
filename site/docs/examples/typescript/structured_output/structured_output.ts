/**
 * Structured Output Example
 *
 * This example demonstrates how to use structured output with Strands Agents to
 * get type-safe, validated responses using Zod schemas.
 */
import { Agent } from '@strands-agents/sdk'
import { z } from 'zod'

async function basicExample(): Promise<void> {
  console.log('\n--- Basic Example ---')

  const PersonInfo = z.object({
    name: z.string(),
    age: z.number(),
    occupation: z.string(),
  })

  const agent = new Agent()
  const result = await agent.invoke('John Smith is a 30-year-old software engineer', {
    structuredOutputSchema: PersonInfo,
  })

  console.log(`Name: ${result.structuredOutput.name}`) // "John Smith"
  console.log(`Age: ${result.structuredOutput.age}`) // 30
  console.log(`Job: ${result.structuredOutput.occupation}`) // "software engineer"
}

async function complexNestedSchemaExample(): Promise<void> {
  console.log('\n--- Complex Nested Schema Example ---')

  const Address = z.object({
    street: z.string(),
    city: z.string(),
    country: z.string(),
    postalCode: z.string().optional(),
  })

  const Contact = z.object({
    email: z.string().optional(),
    phone: z.string().optional(),
  })

  const Person = z.object({
    name: z.string().describe('Full name of the person'),
    age: z.number().describe('Age in years'),
    address: Address.describe('Home address'),
    contacts: z.array(Contact).describe('Contact methods'),
    skills: z.array(z.string()).describe('Professional skills'),
  })

  const agent = new Agent()
  const result = await agent.invoke(
    'Extract info: Jane Doe, a systems admin, 28, lives at 123 Main St, New York, USA. Email: jane@example.com',
    { structuredOutputSchema: Person },
  )

  console.log(`Name: ${result.structuredOutput.name}`) // "Jane Doe"
  console.log(`Age: ${result.structuredOutput.age}`) // 28
  console.log(`Street: ${result.structuredOutput.address.street}`) // "123 Main St"
  console.log(`City: ${result.structuredOutput.address.city}`) // "New York"
  console.log(`Country: ${result.structuredOutput.address.country}`) // "USA"
  console.log(`Email: ${result.structuredOutput.contacts[0].email}`) // "jane@example.com"
  console.log(`Skills: ${result.structuredOutput.skills}`) // ["systems admin"]
}

async function main(): Promise<void> {
  console.log('Structured Output Examples\n')

  await basicExample()
  await complexNestedSchemaExample()

  console.log('\nExamples completed.')
}

main()
