class TAException(Exception):
    """Enhanced exception for Technical Analysis operations"""
    def __init__(self, message: str, error_code: str = "GENERAL"):
        self.error_code = error_code
        super().__init__(message)
