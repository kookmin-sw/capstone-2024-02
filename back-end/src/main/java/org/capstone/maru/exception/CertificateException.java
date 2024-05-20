package org.capstone.maru.exception;

import lombok.Getter;

@Getter
public class CertificateException extends RuntimeException {

    private final RestErrorCode errorCode;
    private final String reason;

    public CertificateException(RestErrorCode errorCode) {
        this(errorCode, null);
    }

    public CertificateException(RestErrorCode errorCode, String reason) {
        this.errorCode = errorCode;
        this.reason = reason;
    }
}
