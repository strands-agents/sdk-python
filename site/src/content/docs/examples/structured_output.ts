import { Agent } from '@strands-agents/sdk'
import { z } from 'zod'

// --8<-- [start:basic_example]
const PersonInfo = z.object({
  name: z.string().describe('Name of the person'),
  age: z.number().describe('Age of the person'),
  occupation: z.string().describe('Occupation of the person'),
})

type PersonInfo = z.infer<typeof PersonInfo>

const basicAgent = new Agent()
const basicResult = await basicAgent.invoke('John Smith is a 30-year-old software engineer', {
  structuredOutputSchema: PersonInfo,
})

const person = basicResult.structuredOutput as PersonInfo
console.log(`Name: ${person.name}`) // "John Smith"
console.log(`Age: ${person.age}`) // 30
console.log(`Job: ${person.occupation}`) // "software engineer"
// --8<-- [end:basic_example]

async function nestedExample() {
  // --8<-- [start:nested_models]
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

  type Person = z.infer<typeof Person>

  const agent = new Agent()
  const result = await agent.invoke(
    'Extract info: Jane Doe, a systems admin, 28, lives at 123 Main St, New York, USA. Email: jane@example.com',
    { structuredOutputSchema: Person },
  )

  const person = result.structuredOutput as Person
  console.log(`Name: ${person.name}`) // "Jane Doe"
  console.log(`Age: ${person.age}`) // 28
  console.log(`Street: ${person.address.street}`) // "123 Main St"
  console.log(`City: ${person.address.city}`) // "New York"
  console.log(`Email: ${person.contacts[0].email}`) // "jane@example.com"
  // --8<-- [end:nested_models]
}
