class OpenSceneSenseError(Exception):
    """Base exception for all expected OpenSceneSense failures."""


class ConfigurationError(OpenSceneSenseError, ValueError):
    pass


class MissingDependencyError(OpenSceneSenseError, ImportError):
    pass


class VideoLoadError(OpenSceneSenseError):
    pass


class VideoMetadataError(OpenSceneSenseError):
    pass


class AudioExtractionError(OpenSceneSenseError):
    pass


class TranscriptionError(OpenSceneSenseError):
    pass


class ProviderConnectionError(OpenSceneSenseError):
    pass


class AuthenticationError(OpenSceneSenseError):
    pass


class RateLimitError(OpenSceneSenseError):
    pass


class ModelNotFoundError(OpenSceneSenseError):
    pass


class ModelCapabilityError(OpenSceneSenseError):
    pass


class ResponseValidationError(OpenSceneSenseError):
    pass


# v1.1 compatibility names.
VideoAnalyzerError = OpenSceneSenseError
VideoAnalysisError = VideoLoadError
AudioTranscriptionError = TranscriptionError
FrameAnalysisError = ResponseValidationError
