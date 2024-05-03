package org.capstone.maru.exception;

import lombok.Getter;

@Getter
public class PostNotFoundException extends RuntimeException {

    private final RestErrorCode errorCode;
    private final String reason;

    public PostNotFoundException(RestErrorCode errorCode) {
        this(errorCode, null);
    }

    public PostNotFoundException(RestErrorCode errorCode, String reason) {
        this.errorCode = errorCode;
        this.reason = reason;
    }
}
