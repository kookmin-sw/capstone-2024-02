package org.capstone.maru.security.exception;

import org.capstone.maru.exception.RestErrorCode;
import org.springframework.security.core.AuthenticationException;

public class MemberAccountNotFoundException extends AuthenticationException {

    private final RestErrorCode errorCode;
    private final String reason;

    public MemberAccountNotFoundException(String msg, Throwable cause) {
        super(msg, cause);
        this.errorCode = RestErrorCode.NOT_FOUND;
        this.reason = "가입한 회원 계정이 존재하지 않습니다.";
    }

    public MemberAccountNotFoundException(String msg) {
        super(msg);
        this.errorCode = RestErrorCode.NOT_FOUND;
        this.reason = "가입한 회원 계정이 존재하지 않습니다.";
    }

    public MemberAccountNotFoundException(RestErrorCode errorCode) {
        super(errorCode.getMessage());
        this.errorCode = errorCode;
        this.reason = "가입한 회원 계정이 존재하지 않습니다.";
    }

}
