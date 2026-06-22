import { describe, it, expect } from 'vitest'
import { SandboxTimeoutError, SandboxAbortError } from '../errors.js'

describe('sandbox errors', () => {
  describe('SandboxTimeoutError', () => {
    it('builds a message from the timeout duration', () => {
      const error = new SandboxTimeoutError(30)
      expect(error.message).toBe('Execution timed out after 30 seconds')
    })

    it('has the correct error name', () => {
      const error = new SandboxTimeoutError(30)
      expect(error.name).toBe('SandboxTimeoutError')
    })

    it('is an instance of Error', () => {
      const error = new SandboxTimeoutError(30)
      expect(error).toBeInstanceOf(Error)
    })
  })

  describe('SandboxAbortError', () => {
    it('has a fixed message', () => {
      const error = new SandboxAbortError()
      expect(error.message).toBe('Execution aborted')
    })

    it('has the correct error name', () => {
      const error = new SandboxAbortError()
      expect(error.name).toBe('SandboxAbortError')
    })

    it('is an instance of Error', () => {
      const error = new SandboxAbortError()
      expect(error).toBeInstanceOf(Error)
    })
  })
})
