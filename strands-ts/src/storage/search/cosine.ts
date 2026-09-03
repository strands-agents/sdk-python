/** Computes cosine similarity between two vectors. Returns 0 when either vector has zero magnitude. */
export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0
  let normA = 0
  let normB = 0
  for (let idx = 0; idx < a.length; idx++) {
    dot += a[idx]! * b[idx]!
    normA += a[idx]! * a[idx]!
    normB += b[idx]! * b[idx]!
  }
  const denominator = Math.sqrt(normA) * Math.sqrt(normB)
  return denominator === 0 ? 0 : dot / denominator
}
