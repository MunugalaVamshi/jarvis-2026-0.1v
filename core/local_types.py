"""Small local data structures used by the Ollama-only pipeline.

This module replaces SDK-only message/config containers. It performs no
network requests and does not import any hosted AI SDK.
"""
from types import SimpleNamespace

class Record:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def model_dump(self, **kwargs):
        return dict(self.__dict__)

class _EnumValues:
    def __init__(self, **values):
        for key, value in values.items(): setattr(self, key, value)

types = SimpleNamespace(
    LiveConnectConfig=Record, FunctionResponse=Record, Blob=Record,
    SessionResumptionConfig=Record, ContextWindowCompressionConfig=Record,
    SlidingWindow=Record, SpeechConfig=Record, VoiceConfig=Record,
    PrebuiltVoiceConfig=Record, ProactivityConfig=Record,
    AutomaticActivityDetection=Record, RealtimeInputConfig=Record,
    ThinkingConfig=Record,
    EndSensitivity=_EnumValues(END_SENSITIVITY_HIGH="HIGH", END_SENSITIVITY_LOW="LOW"),
    StartSensitivity=_EnumValues(START_SENSITIVITY_HIGH="HIGH", START_SENSITIVITY_LOW="LOW"),
    MediaResolution=_EnumValues(MEDIA_RESOLUTION_LOW="LOW", MEDIA_RESOLUTION_MEDIUM="MEDIUM", MEDIA_RESOLUTION_HIGH="HIGH"),
)
