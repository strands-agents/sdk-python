import { describe, expect, it } from 'vitest'
import { isHostToolAllowed } from '../tool-configuration.js'

describe('isHostToolAllowed', () => {
  it.each([
    ['c', '[!][c-a]', true],
    ['c', '[c-a]', false],
    [']', '[]]', true],
    ['a', '[!a]', false],
    ['b', '[!a]', true],
    ['-', '[-]', true],
    ['a', '[a-]', true],
    ['-', '[a-]', true],
    ['[', '[', true],
    ['a', '[abc', false],
    ['get_weather', 'get_*', true],
    ['get_weather', 'get_?eather', true],
    ['get_weather', 'GET_*', false],
    ['abc', 'a**c', true],
    ['abc', 'a[!d]c', true],
    ['abc', 'a[^d]c', false],
    ['abc', 'a[!b]c', false],
  ] as const)('matches Python fnmatchcase(%j, %j) as %j', (value, pattern, expected) => {
    expect(isHostToolAllowed(value, [`@${pattern}`])).toBe(expected)
  })
})
