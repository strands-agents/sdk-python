import { describe, expectTypeOf, it } from 'vitest'

import { ModalSandbox } from '../modal.js'

type ModalCommonJsSandbox = import('modal', { with: { 'resolution-mode': 'require' } }).Sandbox

describe('ModalSandbox types', () => {
  it('accepts Modal Sandbox instances from ESM and CommonJS consumers', () => {
    expectTypeOf<import('modal').Sandbox>().toMatchTypeOf<ConstructorParameters<typeof ModalSandbox>[0]['sandbox']>()
    expectTypeOf<ModalCommonJsSandbox>().toMatchTypeOf<ConstructorParameters<typeof ModalSandbox>[0]['sandbox']>()
  })
})
