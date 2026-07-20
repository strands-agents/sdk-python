// Python/TypeScript marks shared with the changelog page.
export { ICONS as LANG_ICONS } from '../changelog/icons'

/**
 * Deterministic hue for an entry's tile, derived from its name so every build
 * renders the same color without storing one per entry. djb2 over the name,
 * folded to 0-359.
 */
export function tileHue(name: string): number {
  let h = 5381
  for (let i = 0; i < name.length; i++) h = (h * 33) ^ name.charCodeAt(i)
  return Math.abs(h) % 360
}
