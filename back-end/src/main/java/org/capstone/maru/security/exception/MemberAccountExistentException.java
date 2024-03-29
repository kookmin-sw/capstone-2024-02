package org.capstone.maru.security.exception;

import lombok.Getter;
import org.capstone.maru.exception.RestErrorCode;
import org.springframework.security.core.AuthenticationException;

@Getter
public class MemberAccountExistentException extends AuthenticationException {

    private final RestErrorCode errorCode;
    private final String reason;

    public MemberAccountExistentException(String msg, Throwable cause) {
        this(RestErrorCode.DUPLICATE_VALUE, msg);
        cause.fillInStackTrace();
    }

    public MemberAccountExistentException(String msg) {
        this(RestErrorCode.DUPLICATE_VALUE, msg);
    }

    public MemberAccountExistentException(RestErrorCode errorCode) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
        this.reason = "이미 가입한 회원 계정이 존재합니다.";
    }

    public MemberAccountExistentException(RestErrorCode errorCode, String reason) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
        this.reason = reason;
    }

}
