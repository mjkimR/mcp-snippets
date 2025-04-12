class NonRetryableError(Exception):
    @property
    def should_not_retry(self):
        return True


class ExternalServiceUnavailableError(NonRetryableError):
    pass
