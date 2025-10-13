 Live API - WebSockets API reference

Preview: The Live API is in preview.

The Live API is a stateful API that uses WebSockets. In this section, you'll find additional details regarding the WebSockets API.
Sessions

A WebSocket connection establishes a session between the client and the Gemini server. After a client initiates a new connection the session can exchange messages with the server to:

    Send text, audio, or video to the Gemini server.
    Receive audio, text, or function call requests from the Gemini server.

WebSocket connection

To start a session, connect to this websocket endpoint:

wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent

Note: The URL is for version v1beta.
Session configuration

The initial message after connection sets the session configuration, which includes the model, generation parameters, system instructions, and tools.

You can change the configuration parameters except the model during the session.

See the following example configuration. Note that the name casing in SDKs may vary. You can look up the Python SDK configuration options here.


{
  "model": string,
  "generationConfig": {
    "candidateCount": integer,
    "maxOutputTokens": integer,
    "temperature": number,
    "topP": number,
    "topK": integer,
    "presencePenalty": number,
    "frequencyPenalty": number,
    "responseModalities": [string],
    "speechConfig": object,
    "mediaResolution": object
  },
  "systemInstruction": string,
  "tools": [object]
}

For more information on the API field, see generationConfig.
Send messages

To exchange messages over the WebSocket connection, the client must send a JSON object over an open WebSocket connection. The JSON object must have exactly one of the fields from the following object set:


{
  "setup": BidiGenerateContentSetup,
  "clientContent": BidiGenerateContentClientContent,
  "realtimeInput": BidiGenerateContentRealtimeInput,
  "toolResponse": BidiGenerateContentToolResponse
}

Supported client messages

See the supported client messages in the following table:
Message 	Description
BidiGenerateContentSetup 	Session configuration to be sent in the first message
BidiGenerateContentClientContent 	Incremental content update of the current conversation delivered from the client
BidiGenerateContentRealtimeInput 	Real time audio, video, or text input
BidiGenerateContentToolResponse 	Response to a ToolCallMessage received from the server
Receive messages

To receive messages from Gemini, listen for the WebSocket 'message' event, and then parse the result according to the definition of the supported server messages.

See the following:

async with client.aio.live.connect(model='...', config=config) as session:
    await session.send(input='Hello world!', end_of_turn=True)
    async for message in session.receive():
        print(message)

Server messages may have a usageMetadata field but will otherwise include exactly one of the other fields from the BidiGenerateContentServerMessage message. (The messageType union is not expressed in JSON so the field will appear at the top-level of the message.)
Messages and events
ActivityEnd

This type has no fields.

Marks the end of user activity.
ActivityHandling

The different ways of handling user activity.
Enums
ACTIVITY_HANDLING_UNSPECIFIED 	If unspecified, the default behavior is START_OF_ACTIVITY_INTERRUPTS.
START_OF_ACTIVITY_INTERRUPTS 	If true, start of activity will interrupt the model's response (also called "barge in"). The model's current response will be cut-off in the moment of the interruption. This is the default behavior.
NO_INTERRUPTION 	The model's response will not be interrupted.
ActivityStart

This type has no fields.

Marks the start of user activity.
AudioTranscriptionConfig

This type has no fields.

The audio transcription configuration.
AutomaticActivityDetection

Configures automatic detection of activity.
Fields
disabled 	

bool

Optional. If enabled (the default), detected voice and text input count as activity. If disabled, the client must send activity signals.
startOfSpeechSensitivity 	

StartSensitivity

Optional. Determines how likely speech is to be detected.
prefixPaddingMs 	

int32

Optional. The required duration of detected speech before start-of-speech is committed. The lower this value, the more sensitive the start-of-speech detection is and shorter speech can be recognized. However, this also increases the probability of false positives.
endOfSpeechSensitivity 	

EndSensitivity

Optional. Determines how likely detected speech is ended.
silenceDurationMs 	

int32

Optional. The required duration of detected non-speech (e.g. silence) before end-of-speech is committed. The larger this value, the longer speech gaps can be without interrupting the user's activity but this will increase the model's latency.
BidiGenerateContentClientContent

Incremental update of the current conversation delivered from the client. All of the content here is unconditionally appended to the conversation history and used as part of the prompt to the model to generate content.

A message here will interrupt any current model generation.
Fields
turns[] 	

Content

Optional. The content appended to the current conversation with the model.

For single-turn queries, this is a single instance. For multi-turn queries, this is a repeated field that contains conversation history and the latest request.
turnComplete 	

bool

Optional. If true, indicates that the server content generation should start with the currently accumulated prompt. Otherwise, the server awaits additional messages before starting generation.
BidiGenerateContentRealtimeInput

User input that is sent in real time.

The different modalities (audio, video and text) are handled as concurrent streams. The ordering across these streams is not guaranteed.

This is different from BidiGenerateContentClientContent in a few ways:

    Can be sent continuously without interruption to model generation.
    If there is a need to mix data interleaved across the BidiGenerateContentClientContent and the BidiGenerateContentRealtimeInput, the server attempts to optimize for best response, but there are no guarantees.
    End of turn is not explicitly specified, but is rather derived from user activity (for example, end of speech).
    Even before the end of turn, the data is processed incrementally to optimize for a fast start of the response from the model.

Fields
mediaChunks[] 	

Blob

Optional. Inlined bytes data for media input. Multiple mediaChunks are not supported, all but the first will be ignored.

DEPRECATED: Use one of audio, video, or text instead.
audio 	

Blob

Optional. These form the realtime audio input stream.
video 	

Blob

Optional. These form the realtime video input stream.
activityStart 	

ActivityStart

Optional. Marks the start of user activity. This can only be sent if automatic (i.e. server-side) activity detection is disabled.
activityEnd 	

ActivityEnd

Optional. Marks the end of user activity. This can only be sent if automatic (i.e. server-side) activity detection is disabled.
audioStreamEnd 	

bool

Optional. Indicates that the audio stream has ended, e.g. because the microphone was turned off.

This should only be sent when automatic activity detection is enabled (which is the default).

The client can reopen the stream by sending an audio message.
text 	

string

Optional. These form the realtime text input stream.
BidiGenerateContentServerContent

Incremental server update generated by the model in response to client messages.

Content is generated as quickly as possible, and not in real time. Clients may choose to buffer and play it out in real time.
Fields
generationComplete 	

bool

Output only. If true, indicates that the model is done generating.

When model is interrupted while generating there will be no 'generation_complete' message in interrupted turn, it will go through 'interrupted > turn_complete'.

When model assumes realtime playback there will be delay between generation_complete and turn_complete that is caused by model waiting for playback to finish.
turnComplete 	

bool

Output only. If true, indicates that the model has completed its turn. Generation will only start in response to additional client messages.
interrupted 	

bool

Output only. If true, indicates that a client message has interrupted current model generation. If the client is playing out the content in real time, this is a good signal to stop and empty the current playback queue.
groundingMetadata 	

GroundingMetadata

Output only. Grounding metadata for the generated content.
inputTranscription 	

BidiGenerateContentTranscription

Output only. Input audio transcription. The transcription is sent independently of the other server messages and there is no guaranteed ordering.
outputTranscription 	

BidiGenerateContentTranscription

Output only. Output audio transcription. The transcription is sent independently of the other server messages and there is no guaranteed ordering, in particular not between serverContent and this outputTranscription.
urlContextMetadata 	

UrlContextMetadata
modelTurn 	

Content

Output only. The content that the model has generated as part of the current conversation with the user.
BidiGenerateContentServerMessage

Response message for the BidiGenerateContent call.
Fields
usageMetadata 	

UsageMetadata

Output only. Usage metadata about the response(s).
Union field messageType. The type of the message. messageType can be only one of the following:
setupComplete 	

BidiGenerateContentSetupComplete

Output only. Sent in response to a BidiGenerateContentSetup message from the client when setup is complete.
serverContent 	

BidiGenerateContentServerContent

Output only. Content generated by the model in response to client messages.
toolCall 	

BidiGenerateContentToolCall

Output only. Request for the client to execute the functionCalls and return the responses with the matching ids.
toolCallCancellation 	

BidiGenerateContentToolCallCancellation

Output only. Notification for the client that a previously issued ToolCallMessage with the specified ids should be cancelled.
goAway 	

GoAway

Output only. A notice that the server will soon disconnect.
sessionResumptionUpdate 	

SessionResumptionUpdate

Output only. Update of the session resumption state.
BidiGenerateContentSetup

Message to be sent in the first (and only in the first) BidiGenerateContentClientMessage. Contains configuration that will apply for the duration of the streaming RPC.

Clients should wait for a BidiGenerateContentSetupComplete message before sending any additional messages.
Fields
model 	

string

Required. The model's resource name. This serves as an ID for the Model to use.

Format: models/{model}
generationConfig 	

GenerationConfig

Optional. Generation config.

The following fields are not supported:

    responseLogprobs
    responseMimeType
    logprobs
    responseSchema
    stopSequence
    routingConfig
    audioTimestamp

systemInstruction 	

Content

Optional. The user provided system instructions for the model.

Note: Only text should be used in parts and content in each part will be in a separate paragraph.
tools[] 	

Tool

Optional. A list of Tools the model may use to generate the next response.

A Tool is a piece of code that enables the system to interact with external systems to perform an action, or set of actions, outside of knowledge and scope of the model.
realtimeInputConfig 	

RealtimeInputConfig

Optional. Configures the handling of realtime input.
sessionResumption 	

SessionResumptionConfig

Optional. Configures session resumption mechanism.

If included, the server will send SessionResumptionUpdate messages.
contextWindowCompression 	

ContextWindowCompressionConfig

Optional. Configures a context window compression mechanism.

If included, the server will automatically reduce the size of the context when it exceeds the configured length.
inputAudioTranscription 	

AudioTranscriptionConfig

Optional. If set, enables transcription of voice input. The transcription aligns with the input audio language, if configured.
outputAudioTranscription 	

AudioTranscriptionConfig

Optional. If set, enables transcription of the model's audio output. The transcription aligns with the language code specified for the output audio, if configured.
proactivity 	

ProactivityConfig

Optional. Configures the proactivity of the model.

This allows the model to respond proactively to the input and to ignore irrelevant input.
BidiGenerateContentSetupComplete

This type has no fields.

Sent in response to a BidiGenerateContentSetup message from the client.
BidiGenerateContentToolCall

Request for the client to execute the functionCalls and return the responses with the matching ids.
Fields
functionCalls[] 	

FunctionCall

Output only. The function call to be executed.
BidiGenerateContentToolCallCancellation

Notification for the client that a previously issued ToolCallMessage with the specified ids should not have been executed and should be cancelled. If there were side-effects to those tool calls, clients may attempt to undo the tool calls. This message occurs only in cases where the clients interrupt server turns.
Fields
ids[] 	

string

Output only. The ids of the tool calls to be cancelled.
BidiGenerateContentToolResponse

Client generated response to a ToolCall received from the server. Individual FunctionResponse objects are matched to the respective FunctionCall objects by the id field.

Note that in the unary and server-streaming GenerateContent APIs function calling happens by exchanging the Content parts, while in the bidi GenerateContent APIs function calling happens over these dedicated set of messages.
Fields
functionResponses[] 	

FunctionResponse

Optional. The response to the function calls.
BidiGenerateContentTranscription

Transcription of audio (input or output).
Fields
text 	

string

Transcription text.
ContextWindowCompressionConfig

Enables context window compression — a mechanism for managing the model's context window so that it does not exceed a given length.
Fields
Union field compressionMechanism. The context window compression mechanism used. compressionMechanism can be only one of the following:
slidingWindow 	

SlidingWindow

A sliding-window mechanism.
triggerTokens 	

int64

The number of tokens (before running a turn) required to trigger a context window compression.

This can be used to balance quality against latency as shorter context windows may result in faster model responses. However, any compression operation will cause a temporary latency increase, so they should not be triggered frequently.

If not set, the default is 80% of the model's context window limit. This leaves 20% for the next user request/model response.
EndSensitivity

Determines how end of speech is detected.
Enums
END_SENSITIVITY_UNSPECIFIED 	The default is END_SENSITIVITY_HIGH.
END_SENSITIVITY_HIGH 	Automatic detection ends speech more often.
END_SENSITIVITY_LOW 	Automatic detection ends speech less often.
GoAway

A notice that the server will soon disconnect.
Fields
timeLeft 	

Duration

The remaining time before the connection will be terminated as ABORTED.

This duration will never be less than a model-specific minimum, which will be specified together with the rate limits for the model.
ProactivityConfig

Config for proactivity features.
Fields
proactiveAudio 	

bool

Optional. If enabled, the model can reject responding to the last prompt. For example, this allows the model to ignore out of context speech or to stay silent if the user did not make a request, yet.
RealtimeInputConfig

Configures the realtime input behavior in BidiGenerateContent.
Fields
automaticActivityDetection 	

AutomaticActivityDetection

Optional. If not set, automatic activity detection is enabled by default. If automatic voice detection is disabled, the client must send activity signals.
activityHandling 	

ActivityHandling

Optional. Defines what effect activity has.
turnCoverage 	

TurnCoverage

Optional. Defines which input is included in the user's turn.
SessionResumptionConfig

Session resumption configuration.

This message is included in the session configuration as BidiGenerateContentSetup.sessionResumption. If configured, the server will send SessionResumptionUpdate messages.
Fields
handle 	

string

The handle of a previous session. If not present then a new session is created.

Session handles come from SessionResumptionUpdate.token values in previous connections.
SessionResumptionUpdate

Update of the session resumption state.

Only sent if BidiGenerateContentSetup.sessionResumption was set.
Fields
newHandle 	

string

New handle that represents a state that can be resumed. Empty if resumable=false.
resumable 	

bool

True if the current session can be resumed at this point.

Resumption is not possible at some points in the session. For example, when the model is executing function calls or generating. Resuming the session (using a previous session token) in such a state will result in some data loss. In these cases, newHandle will be empty and resumable will be false.
SlidingWindow

The SlidingWindow method operates by discarding content at the beginning of the context window. The resulting context will always begin at the start of a USER role turn. System instructions and any BidiGenerateContentSetup.prefixTurns will always remain at the beginning of the result.
Fields
targetTokens 	

int64

The target number of tokens to keep. The default value is trigger_tokens/2.

Discarding parts of the context window causes a temporary latency increase so this value should be calibrated to avoid frequent compression operations.
StartSensitivity

Determines how start of speech is detected.
Enums
START_SENSITIVITY_UNSPECIFIED 	The default is START_SENSITIVITY_HIGH.
START_SENSITIVITY_HIGH 	Automatic detection will detect the start of speech more often.
START_SENSITIVITY_LOW 	Automatic detection will detect the start of speech less often.
TurnCoverage

Options about which input is included in the user's turn.
Enums
TURN_COVERAGE_UNSPECIFIED 	If unspecified, the default behavior is TURN_INCLUDES_ONLY_ACTIVITY.
TURN_INCLUDES_ONLY_ACTIVITY 	The users turn only includes activity since the last turn, excluding inactivity (e.g. silence on the audio stream). This is the default behavior.
TURN_INCLUDES_ALL_INPUT 	The users turn includes all realtime input since the last turn, including inactivity (e.g. silence on the audio stream).
UrlContextMetadata

Metadata related to url context retrieval tool.
Fields
urlMetadata[] 	

UrlMetadata

List of url context.
UsageMetadata

Usage metadata about response(s).
Fields
promptTokenCount 	

int32

Output only. Number of tokens in the prompt. When cachedContent is set, this is still the total effective prompt size meaning this includes the number of tokens in the cached content.
cachedContentTokenCount 	

int32

Number of tokens in the cached part of the prompt (the cached content)
responseTokenCount 	

int32

Output only. Total number of tokens across all the generated response candidates.
toolUsePromptTokenCount 	

int32

Output only. Number of tokens present in tool-use prompt(s).
thoughtsTokenCount 	

int32

Output only. Number of tokens of thoughts for thinking models.
totalTokenCount 	

int32

Output only. Total token count for the generation request (prompt + response candidates).
promptTokensDetails[] 	

ModalityTokenCount

Output only. List of modalities that were processed in the request input.
cacheTokensDetails[] 	

ModalityTokenCount

Output only. List of modalities of the cached content in the request input.
responseTokensDetails[] 	

ModalityTokenCount

Output only. List of modalities that were returned in the response.
toolUsePromptTokensDetails[] 	

ModalityTokenCount

Output only. List of modalities that were processed for tool-use request inputs.
Ephemeral authentication tokens

Ephemeral authentication tokens can be obtained by calling AuthTokenService.CreateToken and then used with GenerativeService.BidiGenerateContentConstrained, either by passing the token in an access_token query parameter, or in an HTTP Authorization header with "Token" prefixed to it.
CreateAuthTokenRequest

Create an ephemeral authentication token.
Fields
authToken 	

AuthToken

Required. The token to create.
AuthToken

A request to create an ephemeral authentication token.
Fields
name 	

string

Output only. Identifier. The token itself.
expireTime 	

Timestamp

Optional. Input only. Immutable. An optional time after which, when using the resulting token, messages in BidiGenerateContent sessions will be rejected. (Gemini may preemptively close the session after this time.)

If not set then this defaults to 30 minutes in the future. If set, this value must be less than 20 hours in the future.
newSessionExpireTime 	

Timestamp

Optional. Input only. Immutable. The time after which new Live API sessions using the token resulting from this request will be rejected.

If not set this defaults to 60 seconds in the future. If set, this value must be less than 20 hours in the future.
fieldMask 	

FieldMask

Optional. Input only. Immutable. If field_mask is empty, and bidiGenerateContentSetup is not present, then the effective BidiGenerateContentSetup message is taken from the Live API connection.

If field_mask is empty, and bidiGenerateContentSetup is present, then the effective BidiGenerateContentSetup message is taken entirely from bidiGenerateContentSetup in this request. The setup message from the Live API connection is ignored.

If field_mask is not empty, then the corresponding fields from bidiGenerateContentSetup will overwrite the fields from the setup message in the Live API connection.
Union field config. The method-specific configuration for the resulting token. config can be only one of the following:
bidiGenerateContentSetup 	

BidiGenerateContentSetup

Optional. Input only. Immutable. Configuration specific to BidiGenerateContent.
uses 	

int32

Optional. Input only. Immutable. The number of times the token can be used. If this value is zero then no limit is applied. Resuming a Live API session does not count as a use. If unspecified, the default is 1.


 Live API capabilities guide

Preview: The Live API is in preview.

This is a comprehensive guide that covers capabilities and configurations available with the Live API. See Get started with Live API page for a overview and sample code for common use cases.
Before you begin

    Familiarize yourself with core concepts: If you haven't already done so, read the Get started with Live API page first. This will introduce you to the fundamental principles of the Live API, how it works, and the distinction between the different models and their corresponding audio generation methods (native audio or half-cascade).
    Try the Live API in AI Studio: You may find it useful to try the Live API in Google AI Studio before you start building. To use the Live API in Google AI Studio, select Stream.

Establishing a connection

The following example shows how to create a connection with an API key:
Python
JavaScript

import asyncio
from google import genai

client = genai.Client()

model = "gemini-live-2.5-flash-preview"
config = {"response_modalities": ["TEXT"]}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        print("Session started")

if __name__ == "__main__":
    asyncio.run(main())

Note: You can only set one modality in the response_modalities field. This means that you can configure the model to respond with either text or audio, but not both in the same session.
Interaction modalities

The following sections provide examples and supporting context for the different input and output modalities available in Live API.
Sending and receiving text

Here's how you can send and receive text:
Python
JavaScript

import asyncio
from google import genai

client = genai.Client()
model = "gemini-live-2.5-flash-preview"

config = {"response_modalities": ["TEXT"]}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        message = "Hello, how are you?"
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": message}]}, turn_complete=True
        )

        async for response in session.receive():
            if response.text is not None:
                print(response.text, end="")

if __name__ == "__main__":
    asyncio.run(main())

Incremental content updates

Use incremental updates to send text input, establish session context, or restore session context. For short contexts you can send turn-by-turn interactions to represent the exact sequence of events:
Python
JavaScript

turns = [
    {"role": "user", "parts": [{"text": "What is the capital of France?"}]},
    {"role": "model", "parts": [{"text": "Paris"}]},
]

await session.send_client_content(turns=turns, turn_complete=False)

turns = [{"role": "user", "parts": [{"text": "What is the capital of Germany?"}]}]

await session.send_client_content(turns=turns, turn_complete=True)

For longer contexts it's recommended to provide a single message summary to free up the context window for subsequent interactions. See Session Resumption for another method for loading session context.
Sending and receiving audio

The most common audio example, audio-to-audio, is covered in the Getting started guide.

Here's an audio-to-text example that reads a WAV file, sends it in the correct format and receives text output:
Python
JavaScript

# Test file: https://storage.googleapis.com/generativeai-downloads/data/16000.wav
# Install helpers for converting files: pip install librosa soundfile
import asyncio
import io
from pathlib import Path
from google import genai
from google.genai import types
import soundfile as sf
import librosa

client = genai.Client()
model = "gemini-live-2.5-flash-preview"

config = {"response_modalities": ["TEXT"]}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:

        buffer = io.BytesIO()
        y, sr = librosa.load("sample.wav", sr=16000)
        sf.write(buffer, y, sr, format='RAW', subtype='PCM_16')
        buffer.seek(0)
        audio_bytes = buffer.read()

        # If already in correct format, you can use this:
        # audio_bytes = Path("sample.pcm").read_bytes()

        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )

        async for response in session.receive():
            if response.text is not None:
                print(response.text)

if __name__ == "__main__":
    asyncio.run(main())

And here is a text-to-audio example. You can receive audio by setting AUDIO as response modality. This example saves the received data as WAV file:
Python
JavaScript

import asyncio
import wave
from google import genai

client = genai.Client()
model = "gemini-live-2.5-flash-preview"

config = {"response_modalities": ["AUDIO"]}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        wf = wave.open("audio.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        message = "Hello how are you?"
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": message}]}, turn_complete=True
        )

        async for response in session.receive():
            if response.data is not None:
                wf.writeframes(response.data)

            # Un-comment this code to print audio data info
            # if response.server_content.model_turn is not None:
            #      print(response.server_content.model_turn.parts[0].inline_data.mime_type)

        wf.close()

if __name__ == "__main__":
    asyncio.run(main())

Audio formats

Audio data in the Live API is always raw, little-endian, 16-bit PCM. Audio output always uses a sample rate of 24kHz. Input audio is natively 16kHz, but the Live API will resample if needed so any sample rate can be sent. To convey the sample rate of input audio, set the MIME type of each audio-containing Blob to a value like audio/pcm;rate=16000.
Audio transcriptions

You can enable transcription of the model's audio output by sending output_audio_transcription in the setup config. The transcription language is inferred from the model's response.
Python
JavaScript

import asyncio
from google import genai
from google.genai import types

client = genai.Client()
model = "gemini-live-2.5-flash-preview"

config = {"response_modalities": ["AUDIO"],
        "output_audio_transcription": {}
}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        message = "Hello? Gemini are you there?"

        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": message}]}, turn_complete=True
        )

        async for response in session.receive():
            if response.server_content.model_turn:
                print("Model turn:", response.server_content.model_turn)
            if response.server_content.output_transcription:
                print("Transcript:", response.server_content.output_transcription.text)

if __name__ == "__main__":
    asyncio.run(main())

You can enable transcription of the audio input by sending input_audio_transcription in setup config.
Python
JavaScript

import asyncio
from pathlib import Path
from google import genai
from google.genai import types

client = genai.Client()
model = "gemini-live-2.5-flash-preview"

config = {
    "response_modalities": ["TEXT"],
    "input_audio_transcription": {},
}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        audio_data = Path("16000.pcm").read_bytes()

        await session.send_realtime_input(
            audio=types.Blob(data=audio_data, mime_type='audio/pcm;rate=16000')
        )

        async for msg in session.receive():
            if msg.server_content.input_transcription:
                print('Transcript:', msg.server_content.input_transcription.text)

if __name__ == "__main__":
    asyncio.run(main())

Stream audio and video

To see an example of how to use the Live API in a streaming audio and video format, run the "Live API - Get Started" file in the cookbooks repository:

View on Colab
Change voice and language

The Live API models each support a different set of voices. Half-cascade supports Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr. Native audio supports a much longer list (identical to the TTS model list). You can listen to all the voices in AI Studio.

To specify a voice, set the voice name within the speechConfig object as part of the session configuration:
Python
JavaScript

config = {
    "response_modalities": ["AUDIO"],
    "speech_config": {
        "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
    },
}

Note: If you're using the generateContent API, the set of available voices is slightly different. See the audio generation guide for generateContent audio generation voices.

The Live API supports multiple languages.

To change the language, set the language code within the speechConfig object as part of the session configuration:
Python
JavaScript

config = {
    "response_modalities": ["AUDIO"],
    "speech_config": {
        "language_code": "de-DE"
    }
}

Note: Native audio output models automatically choose the appropriate language and don't support explicitly setting the language code.
Native audio capabilities

The following capabilities are only available with native audio. You can learn more about native audio in Choose a model and audio generation.
Note: Native audio models currently have limited tool use support. See Overview of supported tools for details.
How to use native audio output

To use native audio output, configure one of the native audio models and set response_modalities to AUDIO.

See Send and receive audio for a full example.
Python
JavaScript

model = "gemini-2.5-flash-native-audio-preview-09-2025"
config = types.LiveConnectConfig(response_modalities=["AUDIO"])

async with client.aio.live.connect(model=model, config=config) as session:
    # Send audio input and receive audio

Affective dialog

This feature lets Gemini adapt its response style to the input expression and tone.

To use affective dialog, set the api version to v1alpha and set enable_affective_dialog to truein the setup message:
Python
JavaScript

client = genai.Client(http_options={"api_version": "v1alpha"})

config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    enable_affective_dialog=True
)

Note that affective dialog is currently only supported by the native audio output models.
Proactive audio

When this feature is enabled, Gemini can proactively decide not to respond if the content is not relevant.

To use it, set the api version to v1alpha and configure the proactivity field in the setup message and set proactive_audio to true:
Python
JavaScript

client = genai.Client(http_options={"api_version": "v1alpha"})

config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    proactivity={'proactive_audio': True}
)

Note that proactive audio is currently only supported by the native audio output models.
Thinking

The latest native audio output model gemini-2.5-flash-native-audio-preview-09-2025 supports thinking capabilities, with dynamic thinking enabled by default.

The thinkingBudget parameter guides the model on the number of thinking tokens to use when generating a response. You can disable thinking by setting thinkingBudget to 0. For more info on the thinkingBudget configuration details of the model, see the thinking budgets documentation.
Python
JavaScript

model = "gemini-2.5-flash-native-audio-preview-09-2025"

config = types.LiveConnectConfig(
    response_modalities=["AUDIO"]
    thinking_config=types.ThinkingConfig(
        thinking_budget=1024,
    )
)

async with client.aio.live.connect(model=model, config=config) as session:
    # Send audio input and receive audio

Additionally, you can enable thought summaries by setting includeThoughts to true in your configuration. See thought summaries for more info:
Python
JavaScript

model = "gemini-2.5-flash-native-audio-preview-09-2025"

config = types.LiveConnectConfig(
    response_modalities=["AUDIO"]
    thinking_config=types.ThinkingConfig(
        thinking_budget=1024,
        include_thoughts=True
    )
)

Voice Activity Detection (VAD)

Voice Activity Detection (VAD) allows the model to recognize when a person is speaking. This is essential for creating natural conversations, as it allows a user to interrupt the model at any time.

When VAD detects an interruption, the ongoing generation is canceled and discarded. Only the information already sent to the client is retained in the session history. The server then sends a BidiGenerateContentServerContent message to report the interruption.

The Gemini server then discards any pending function calls and sends a BidiGenerateContentServerContent message with the IDs of the canceled calls.
Python
JavaScript

async for response in session.receive():
    if response.server_content.interrupted is True:
        # The generation was interrupted

        # If realtime playback is implemented in your application,
        # you should stop playing audio and clear queued playback here.

Automatic VAD

By default, the model automatically performs VAD on a continuous audio input stream. VAD can be configured with the realtimeInputConfig.automaticActivityDetection field of the setup configuration.

When the audio stream is paused for more than a second (for example, because the user switched off the microphone), an audioStreamEnd event should be sent to flush any cached audio. The client can resume sending audio data at any time.
Python
JavaScript

# example audio file to try:
# URL = "https://storage.googleapis.com/generativeai-downloads/data/hello_are_you_there.pcm"
# !wget -q $URL -O sample.pcm
import asyncio
from pathlib import Path
from google import genai
from google.genai import types

client = genai.Client()
model = "gemini-live-2.5-flash-preview"

config = {"response_modalities": ["TEXT"]}

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        audio_bytes = Path("sample.pcm").read_bytes()

        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )

        # if stream gets paused, send:
        # await session.send_realtime_input(audio_stream_end=True)

        async for response in session.receive():
            if response.text is not None:
                print(response.text)

if __name__ == "__main__":
    asyncio.run(main())

With send_realtime_input, the API will respond to audio automatically based on VAD. While send_client_content adds messages to the model context in order, send_realtime_input is optimized for responsiveness at the expense of deterministic ordering.
Automatic VAD configuration

For more control over the VAD activity, you can configure the following parameters. See API reference for more info.
Python
JavaScript

from google.genai import types

config = {
    "response_modalities": ["TEXT"],
    "realtime_input_config": {
        "automatic_activity_detection": {
            "disabled": False, # default
            "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_LOW,
            "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_LOW,
            "prefix_padding_ms": 20,
            "silence_duration_ms": 100,
        }
    }
}

Disable automatic VAD

Alternatively, the automatic VAD can be disabled by setting realtimeInputConfig.automaticActivityDetection.disabled to true in the setup message. In this configuration the client is responsible for detecting user speech and sending activityStart and activityEnd messages at the appropriate times. An audioStreamEnd isn't sent in this configuration. Instead, any interruption of the stream is marked by an activityEnd message.
Python
JavaScript

config = {
    "response_modalities": ["TEXT"],
    "realtime_input_config": {"automatic_activity_detection": {"disabled": True}},
}

async with client.aio.live.connect(model=model, config=config) as session:
    # ...
    await session.send_realtime_input(activity_start=types.ActivityStart())
    await session.send_realtime_input(
        audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
    )
    await session.send_realtime_input(activity_end=types.ActivityEnd())
    # ...

Token count

You can find the total number of consumed tokens in the usageMetadata field of the returned server message.
Python
JavaScript

async for message in session.receive():
    # The server will periodically send messages that include UsageMetadata.
    if message.usage_metadata:
        usage = message.usage_metadata
        print(
            f"Used {usage.total_token_count} tokens in total. Response token breakdown:"
        )
        for detail in usage.response_tokens_details:
            match detail:
                case types.ModalityTokenCount(modality=modality, token_count=count):
                    print(f"{modality}: {count}")

Media resolution

You can specify the media resolution for the input media by setting the mediaResolution field as part of the session configuration:
Python
JavaScript

from google.genai import types

config = {
    "response_modalities": ["AUDIO"],
    "media_resolution": types.MediaResolution.MEDIA_RESOLUTION_LOW,
}

Limitations

Consider the following limitations of the Live API when you plan your project.
Response modalities

You can only set one response modality (TEXT or AUDIO) per session in the session configuration. Setting both results in a config error message. This means that you can configure the model to respond with either text or audio, but not both in the same session.
Client authentication

The Live API only provides server-to-server authentication by default. If you're implementing your Live API application using a client-to-server approach, you need to use ephemeral tokens to mitigate security risks.
Session duration

Audio-only sessions are limited to 15 minutes, and audio plus video sessions are limited to 2 minutes. However, you can configure different session management techniques for unlimited extensions on session duration.
Context window

A session has a context window limit of:

    128k tokens for native audio output models
    32k tokens for other Live API models

Supported languages

Live API supports the following languages.
Note: Native audio output models automatically choose the appropriate language and don't support explicitly setting the language code.
Language 	BCP-47 Code 	Language 	BCP-47 Code
German (Germany) 	de-DE 	English (Australia)* 	en-AU
English (UK)* 	en-GB 	English (India) 	en-IN
English (US) 	en-US 	Spanish (US) 	es-US
French (France) 	fr-FR 	Hindi (India) 	hi-IN
Portuguese (Brazil) 	pt-BR 	Arabic (Generic) 	ar-XA
Spanish (Spain)* 	es-ES 	French (Canada)* 	fr-CA
Indonesian (Indonesia) 	id-ID 	Italian (Italy) 	it-IT
Japanese (Japan) 	ja-JP 	Turkish (Turkey) 	tr-TR
Vietnamese (Vietnam) 	vi-VN 	Bengali (India) 	bn-IN
Gujarati (India)* 	gu-IN 	Kannada (India)* 	kn-IN
Marathi (India) 	mr-IN 	Malayalam (India)* 	ml-IN
Tamil (India) 	ta-IN 	Telugu (India) 	te-IN
Dutch (Netherlands) 	nl-NL 	Korean (South Korea) 	ko-KR
Mandarin Chinese (China)* 	cmn-CN 	Polish (Poland) 	pl-PL
Russian (Russia) 	ru-RU 	Thai (Thailand) 	th-TH

Languages marked with an asterisk (*) are not available for Native audio.