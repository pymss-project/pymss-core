class APIError(Exception):
    def __init__(self, status_code, code, message, param=None, error_type="invalid_request_error"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.param = param
        self.error_type = error_type
