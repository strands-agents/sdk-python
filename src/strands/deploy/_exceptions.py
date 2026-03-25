"""Exceptions for agent deployment operations."""


class DeployException(Exception):
    """Base exception for all deployment operations."""

    pass


class DeployTargetException(DeployException):
    """Raised when a deployment target fails."""

    def __init__(self, target: str, message: str, cause: Exception | None = None):
        self.target = target
        self.cause = cause
        super().__init__(f"Deploy target '{target}' failed: {message}")
        if cause:
            self.__cause__ = cause


class DeployPackagingException(DeployException):
    """Raised when packaging the agent code fails."""

    pass


class DeployStateException(DeployException):
    """Raised when state management operations fail."""

    pass


