When the Amazon Nova Sonic model responds, it follows a structured event sequence. The flow begins with a completionStart event that contains unique identifiers like sessionId, promptName, and completionId. These identifiers are consistent throughout the response cycle and unite all subsequent response events.

Each response type follows a consistent three-part pattern: contentStart defines the content type and format, the actual content event, and contentEnd closes that segment. The response typically includes multiple content blocks in sequence: automatic speech recognition (ASR) transcription (what the user said), optional tool use (when external information is needed), text response (what the model plans to say), and audio response (the spoken output).

The ASR transcription appears first, delivering the model's understanding of the user's speech with role: "USER" and "additionalModelFields": "{\"generationStage\":\"FINAL\"}" in the contentStart. When the model needs external data, it sends tool-related events with specific tool names and parameters. The text response provides a preview of the planned speech with role: "ASSISTANT" and "additionalModelFields": "{\"generationStage\":\"SPECULATIVE\"}". The audio response then delivers base64-encoded speech chunks sharing the same contentId throughout the stream.

During audio generation, Amazon Nova Sonic supports natural conversation flow through its barge-in capability. When a user interrupts Amazon Nova Sonic while it's speaking, Nova Sonic immediately stops generating speech, switches to listening mode, and sends a content notification indicating the interruption has occurred. Because Nova Sonic operates faster than real-time, some audio may have already been delivered but not yet played. The interruption notification enables the client application to clear its audio queue and stop playback immediately, creating a responsive conversational experience.

After audio generation completes (or is interrupted via barge-in), Amazon Nova Sonic provides an additional text response that contains a sentence-level transcription of what was actually spoken. This text response includes a contentStart event with role: "ASSISTANT" and "additionalModelFields": "{\"generationStage\":\"FINAL\"}".

Throughout the response handling, usageEvent events are sent to track token consumption. These events contain detailed metrics on input tokens and output tokens (both speech and text), and their cumulative totals. Each usageEvent maintains the same sessionId, promptName, and completionId as other events in the conversation flow. The details section provides both incremental changes (delta) and running totals of token usage, enabling precise monitoring of the usage during the conversation.

The model sends a completionEnd event with the original identifiers and a stopReason that indicates how the conversation ended. This event hierarchy ensures your application can track which parts of the response belong together and process them accordingly, maintaining conversation context throughout multiple turns.

The output event flow begins by entering the response generation phase. It starts with automatic speech recognition, selects a tool for use, transcribes speech, generates audio, finalizes the transcription, and finishes the session.
Diagram that explains the Amazon Nova Sonic output event flow.
Output event flow

The structure of the output event flow is described in this section.

    UsageEvent

"event": {
    "usageEvent": {
        "completionId": "string", // unique identifier for completion
        "details": {
            "delta": { // incremental changes since last event
                "input": {
                    "speechTokens": number, // input speech tokens
                    "textTokens": number // input text tokens
                },
                "output": {
                    "speechTokens": number, // speech tokens generated
                    "textTokens": number // text tokens generated
                }
            },
            "total": { // cumulative counts
                "input": {
                    "speechTokens": number, // total speech tokens processed
                    "textTokens": number // total text tokens processed
                },
                "output": {
                    "speechTokens": number, // total speech tokens generated
                    "textTokens": number // total text tokens generated
                }
            }
        },
        "promptName": "string", // same unique identifier from promptStart event
        "sessionId": "string", // unique identifier
        "totalInputTokens": number, // cumulative input tokens
        "totalOutputTokens": number, // cumulative output tokens
        "totalTokens": number // total tokens in the session
    }
}

CompleteStartEvent

"event": {
        "completionStart": {
            "sessionId": "string", // unique identifier
            "promptName": "string", // same unique identifier from promptStart event
            "completionId": "string", // unique identifier
        }
    }

TextOutputContent

    ContentStart

"event": {
        "contentStart": {
            "additionalModelFields": "{\"generationStage\":\"FINAL\"}" | "{\"generationStage\":\"SPECULATIVE\"}",
            "sessionId": "string", // unique identifier
            "promptName": "string", // same unique identifier from promptStart event
            "completionId": "string", // unique identifier
            "contentId": "string", // unique identifier for the content block
            "type": "TEXT",
            "role": "USER" | "ASSISTANT",
            "textOutputConfiguration": {
                "mediaType": "text/plain"
            }
        }
    }

TextOutput

"event": {
        "textOutput": {
            "sessionId": "string", // unique identifier
            "promptName": "string", // same unique identifier from promptStart event
            "completionId": "string", // unique identifier
            "contentId": "string", // same unique identifier from its contentStart
            "content": "string" // User transcribe or Text Response
        }
    }

ContentEnd

    "event": {
        "contentEnd": {
                "sessionId": "string", // unique identifier
                "promptName": "string", // same unique identifier from promptStart event
                "completionId": "string", // unique identifier
                "contentId": "string", // same unique identifier from its contentStart
                "stopReason": "PARTIAL_TURN" | "END_TURN" | "INTERRUPTED",
                "type": "TEXT"
        }
      }

ToolUse

    ContentStart

"event": {
    "contentStart": {
      "sessionId": "string", // unique identifier
      "promptName": "string", // same unique identifier from promptStart event
      "completionId": "string", // unique identifier
      "contentId": "string", // unique identifier for the content block
      "type": "TOOL",
      "role": "TOOL",
      "toolUseOutputConfiguration": {
        "mediaType": "application/json"
      }
    }
  }

ToolUse

"event": {
    "toolUse": {
      "sessionId": "string", // unique identifier
      "promptName": "string", // same unique identifier from promptStart event
      "completionId": "string", // unique identifier
      "contentId": "string", // same unique identifier from its contentStart
      "content": "json",
      "toolName": "string",
      "toolUseId": "string"
    }
  }

ContentEnd

    "event": {
        "contentEnd": {
          "sessionId": "string", // unique identifier
          "promptName": "string", // same unique identifier from promptStart event
          "completionId": "string", // unique identifier
          "contentId": "string", // same unique identifier from its contentStart
          "stopReason": "TOOL_USE",
          "type": "TOOL"
        }
      }

AudioOutputContent

    ContentStart

"event": {
    "contentStart": {
      "sessionId": "string", // unique identifier
      "promptName": "string", // same unique identifier from promptStart event
      "completionId": "string", // unique identifier
      "contentId": "string", // unique identifier for the content block
      "type": "AUDIO",
      "role": "ASSISTANT",
      "audioOutputConfiguration": {
            "mediaType": "audio/lpcm",
            "sampleRateHertz": 8000 | 16000 | 24000,
            "sampleSizeBits": 16,
            "encoding": "base64",
            "channelCount": 1
            }
      }
  }

AudioOutput

"event": {
        "audioOutput": {
            "sessionId": "string", // unique identifier
            "promptName": "string", // same unique identifier from promptStart event
            "completionId": "string", // unique identifier
            "contentId": "string", // same unique identifier from its contentStart
            "content": "base64EncodedAudioData", // Audio
        }
    }

ContentEnd

    "event": {
        "contentEnd": {
          "sessionId": "string", // unique identifier
          "promptName": "string", // same unique identifier from promptStart event
          "completionId": "string", // unique identifier
          "contentId": "string", // same unique identifier from its contentStart
          "stopReason": "PARTIAL_TURN" | "END_TURN",
          "type": "AUDIO"
        }
      }

CompletionEndEvent

"event": {
    "completionEnd": {
      "sessionId": "string", // unique identifier
      "promptName": "string", // same unique identifier from promptStart event
      "completionId": "string", // unique identifier
      "stopReason": "END_TURN" 
    }
  }



  The bidirectional Stream API uses an event-driven architecture with structured input and output events. Understanding the correct event ordering is crucial for implementing successful conversational applications and maintaining the proper conversation state throughout interactions.

The Nova Sonic conversation follows a structured event sequence. You begin by sending a sessionStart event that contains the inference configuration parameters, such as temperature and token limits. Next, you send promptStart to define the audio output format and tool configurations, assigning a unique promptName identifier that must be included in all subsequent events.

For each interaction type (system prompt, audio, and so on), you follow a three-part pattern: use contentStart to define the content type and the role of the content (SYSTEM, USER, ASSISTANT, TOOL), then provide the actual content event, and finish with contentEnd to close that segment. The contentStart event specifies whether you're sending tool results, streaming audio, or a system prompt. The contentStart event includes a unique contentName identifier.

A conversation history can be included only once, after the system prompt and before audio streaming begins. It follows the same contentStart/textInput/contentEnd pattern. The USER and ASSISTANT roles must be defined in the contentStart event for each historical message. This provides essential context for the current conversation but must be completed before any new user input begins.

Audio streaming operates with continuous microphone sampling. After sending an initial contentStart, audio frames (approximately 32ms each) are captured directly from the microphone and immediately sent as audioInput events using the same contentName. These audio samples should be streamed in real-time as they're captured, maintaining the natural microphone sampling cadence throughout the conversation. All audio frames share a single content container until the conversation ends and it is explicitly closed.

After the conversation ends or needs to be terminated, it's essential to properly close all open streams and end the session in the correct sequence. To properly end a session and avoid resource leaks, you must follow a specific closing sequence:

    Close any open audio streams with the contentEnd event.

    Send a promptEnd event that references the original promptName.

    Send the sessionEnd event.

Skipping any of these closing events can result in incomplete conversations or orphaned resources.

These identifiers create a hierarchical structure: the promptName ties all conversation events together, while each contentName marks the boundaries of specific content blocks. This hierarchy ensures that model maintains proper context throughout the interaction.
Diagram that explains the Amazon Nova Sonic input event flow.
Input event flow

The structure of the input event flow is provided in this section.

    RequestStartEvent

{
    "event": {
        "sessionStart": {
            "inferenceConfiguration": {
                "maxTokens": "int",
                "topP": "float",
                "temperature": "float"
            }
        }
    }
}

PromptStartEvent

{
    "event": {
        "promptStart": {
            "promptName": "string", // unique identifier same across all events i.e. UUID
            "textOutputConfiguration": {
                "mediaType": "text/plain"
            },
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 8000 | 16000 | 24000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": "matthew" | "tiffany" | "amy" |
                        "lupe" | "carlos" | "ambre" | "florian" |
                        "greta" | "lennart" | "beatrice" | "lorenzo",
                "encoding": "base64",
                "audioType": "SPEECH",
            },
            "toolUseOutputConfiguration": {
                "mediaType": "application/json"
            },
            "toolConfiguration": {
                "tools": [{
                    "toolSpec": {
                        "name": "string",
                        "description": "string",
                        "inputSchema": {
                            "json": "{}"
                        }
                    }
                }]
            }
        }
    }
}

InputContentStartEvent

    Text

{
    "event": {
        "contentStart": {
            "promptName": "string", // same unique identifier from promptStart event
            "contentName": "string", // unique identifier for the content block
            "type": "TEXT",
            "interactive": false,
            "role": "SYSTEM" | "USER" | "ASSISTANT",
            "textInputConfiguration": {
                "mediaType": "text/plain"
            }
        }
    }
}

Audio

{
    "event": {
        "contentStart": {
            "promptName": "string", // same unique identifier from promptStart event
            "contentName": "string", // unique identifier for the content block
            "type": "AUDIO",
            "interactive": true,
            "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 8000 | 16000 | 24000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "audioType": "SPEECH",
                "encoding": "base64"
            }
        }
    }
}

Tool

    {
        "event": {
            "contentStart": {
                "promptName": "string", // same unique identifier from promptStart event
                "contentName": "string", // unique identifier for the content block
                "interactive": false,
                "type": "TOOL",
                "role": "TOOL",
                "toolResultInputConfiguration": {
                    "toolUseId": "string", // existing tool use id
                    "type": "TEXT",
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            }
        }
    }

TextInputContent

{
    "event": {
        "textInput": {
            "promptName": "string", // same unique identifier from promptStart event
            "contentName": "string", // unique identifier for the content block
            "content": "string"
        }
    }
}

AudioInputContent

{
    "event": {
        "audioInput": {
            "promptName": "string", // same unique identifier from promptStart event
            "contentName": "string", // same unique identifier from its contentStart
            "content": "base64EncodedAudioData"
        }
    }
}

ToolResultContentEvent

"event": {
    "toolResult": {
        "promptName": "string", // same unique identifier from promptStart event
        "contentName": "string", // same unique identifier from its contentStart
        "content": "{\"key\": \"value\"}" // stringified JSON object as a tool result 
    }
}

InputContentEndEvent

{
    "event": {
        "contentEnd": {
            "promptName": "string", // same unique identifier from promptStart event
            "contentName": "string" // same unique identifier from its contentStart
        }
    }
}

PromptEndEvent

{
    "event": {
        "promptEnd": {
            "promptName": "string" // same unique identifier from promptStart event
        }
    }
}

RequestEndEvent

{
    "event": {
        "sessionEnd": {}
    }
}