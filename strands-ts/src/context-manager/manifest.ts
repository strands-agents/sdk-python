/**
 * L1 manifest: tracks which messages are stored and their metadata.
 *
 * @internal
 */

/**
 * A single entry in the stash manifest.
 */
export interface ManifestEntry {
  /** The message's durable tracking ID. */
  trackingId: string
  /** Message role (user or assistant). */
  role: string
  /** The key under which the message bytes are stored. */
  storageKey: string
  /** Number of content blocks in the message. */
  contentBlocks: number
}

interface SerializedManifest {
  version: number
  entries: ManifestEntry[]
}

/**
 * The L1 manifest tracks all messages written to the stash.
 *
 * Serialized to JSON and stored alongside the messages. Enables the strategy
 * engine to know what's available in L1 without reading every stored message.
 *
 * @internal
 */
export class Manifest {
  readonly version = 1
  private readonly _entries: ManifestEntry[] = []

  get entries(): readonly ManifestEntry[] {
    return this._entries
  }

  /** Append an entry to the manifest. */
  add(entry: ManifestEntry): void {
    this._entries.push(entry)
  }

  /** Check if a message is already recorded in the manifest. */
  has(trackingId: string): boolean {
    return this._entries.some((entry) => entry.trackingId === trackingId)
  }

  /** Serialize the manifest to JSON bytes. */
  serialize(): Uint8Array {
    const payload: SerializedManifest = { version: this.version, entries: this._entries }
    return new TextEncoder().encode(JSON.stringify(payload))
  }

  /** Deserialize a manifest from JSON bytes. */
  static deserialize(data: Uint8Array): Manifest {
    const payload: SerializedManifest = JSON.parse(new TextDecoder().decode(data))
    const manifest = new Manifest()
    for (const entry of payload.entries ?? []) {
      manifest.add(entry)
    }
    return manifest
  }
}
