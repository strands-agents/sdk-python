const chunks = []

process.stdin.setEncoding('utf8')

for await (const chunk of process.stdin) {
  chunks.push(chunk)
}

const packOutput = JSON.parse(chunks.join(''))
const files = packOutput[0]?.files?.map((file) => file.path) ?? []

const forbiddenPatterns = [
  /(^|\/)__tests__(\/|$)/,
  /(^|\/)__fixtures__(\/|$)/,
  /\.test(?:\.[^/.]+)*\.(?:js|d\.ts)(?:\.map)?$/,
]

const forbiddenFiles = files.filter((file) => forbiddenPatterns.some((pattern) => pattern.test(file)))

if (forbiddenFiles.length > 0) {
  throw new Error(`Published package includes test artifacts:\n${forbiddenFiles.slice(0, 20).join('\n')}`)
}

console.log(`[package-contents] OK (${files.length} files)`)
