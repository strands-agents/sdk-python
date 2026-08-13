import { tool } from '@strands-agents/sdk'
import { z } from 'zod'

export const callCounts = { getWeather: 0, bookFlight: 0 }

/**
 * Creates deterministic tools used by the durable execution example.
 *
 * @returns The tools available to the example agent.
 */
export function buildTools() {
  return [
    tool({
      name: 'get_weather',
      description: 'Get the weather for a city.',
      inputSchema: z.object({ city: z.string() }),
      callback: (input) => {
        callCounts.getWeather += 1
        console.log(`city=<${input.city}>, count=<${callCounts.getWeather}> | get_weather invoked`)
        return `Weather in ${input.city}: 72°F and sunny.`
      },
    }),
    tool({
      name: 'book_flight',
      description: 'Book a flight to a destination.',
      inputSchema: z.object({ destination: z.string() }),
      callback: (input) => {
        callCounts.bookFlight += 1
        console.log(`destination=<${input.destination}>, count=<${callCounts.bookFlight}> | book_flight invoked`)
        return `Flight booked to ${input.destination}: confirmation ABC123.`
      },
    }),
  ]
}
