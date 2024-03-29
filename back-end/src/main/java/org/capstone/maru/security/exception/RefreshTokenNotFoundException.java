package org.capstone.maru.security.exception;

import lombok.Getter;
import org.capstone.maru.exception.RestErrorCode;

@Getter
public class RefreshTokenNotFoundException extends RuntimeException {

    private final RestErrorCode errorCode;
    private final String reason;

    public RefreshTokenNotFoundException() {
        this(RestErrorCode.REFRESH_TOKEN_NOTFOUND);
    }

    public RefreshTokenNotFoundException(String message) {
        this(RestErrorCode.REFRESH_TOKEN_NOTFOUND, message);
    }

    public RefreshTokenNotFoundException(RestErrorCode errorCode) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
        this.reason = "존재하지 않는 리프레시 토큰입니다.";
    }

    public RefreshTokenNotFoundException(RestErrorCode errorCode, String reason) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
        this.reason = reason;
    }
}
