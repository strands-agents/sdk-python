import asyncio
from strands.experimental.bidi import BidiAgent, BidiAudioIO
from strands.experimental.bidi.models.nova_sonic import BidiNovaSonicModel

async def main():
      model = BidiNovaSonicModel()
      audio_io = BidiAudioIO(echo_cancellation=True)

      agent = BidiAgent(
          model=model,
          system_prompt="You are a helpful voice assistant.",
      )

      await agent.run(
          inputs=[audio_io.input()],
          outputs=[audio_io.output()],
      )

asyncio.run(main())