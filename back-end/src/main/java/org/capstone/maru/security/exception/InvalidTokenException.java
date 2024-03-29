package org.capstone.maru.security.exception;

import lombok.Getter;
import org.capstone.maru.exception.RestErrorCode;
import org.springframework.security.core.AuthenticationException;

@Getter
public class InvalidTokenException extends AuthenticationException {

    private final RestErrorCode errorCode;
    private final String reason;

    public InvalidTokenException(String msg, Throwable cause) {
        this(RestErrorCode.INVALID_TOKEN_VALUE, msg);
        cause.fillInStackTrace();
    }

    public InvalidTokenException(String msg) {
        this(RestErrorCode.INVALID_TOKEN_VALUE, msg);
    }

    public InvalidTokenException(RestErrorCode errorCode) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
        this.reason = "만료되었거나 유효 하지 않은 토큰입니다.";
    }

    public InvalidTokenException(RestErrorCode errorCode, String reason) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
        this.reason = reason;
    }
}
