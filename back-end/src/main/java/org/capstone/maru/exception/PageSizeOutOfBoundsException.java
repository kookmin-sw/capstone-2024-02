package org.capstone.maru.exception;

import lombok.Getter;

@Getter
public class PageSizeOutOfBoundsException extends RuntimeException {

    private final RestErrorCode errorCode;
    private final String reason;

    public PageSizeOutOfBoundsException() {
        this.errorCode = RestErrorCode.PAGE_SIZE_OUT_OF_BOUNDS;
        this.reason = null;
    }

    public PageSizeOutOfBoundsException(RestErrorCode errorCode, String reason) {
        this.errorCode = errorCode;
        this.reason = reason;
    }
}
