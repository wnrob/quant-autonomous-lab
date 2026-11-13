class QALError(Exception):
    """Base error for Quant Autonomous Lab."""
class DataError(QALError):
    """Data could not be fetched, validated or loaded."""
class AdapterUnavailable(QALError):
    """A third-party adapter's library is not installed or not compatible."""
class ArtifactIntegrityError(QALError):
    """A stored model artifact failed a path/hash check and was NOT loaded (pickle semantics make loading tampered files code execution)."""
