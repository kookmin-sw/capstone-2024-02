package org.capstone.maru.exception;

import lombok.Getter;

@Getter
public enum RestErrorCode {

    PAGE_SIZE_OUT_OF_BOUNDS(400, "C002", "페이지 사이즈 값이 잘 못되었습니다."),
    UNAUTHORIZED(401, "C001", "인증 되지 않은 사용자입니다."),
    INVALID_TOKEN_VALUE(401, "C002", "유효하지 않은 토큰입니다."),
    REFRESH_TOKEN_NOTFOUND(401, "C003", "존재하지 않는 리프레시 토큰입니다."),
    FORBIDDEN(403, "C001", "RuntimeException 입니다. 백엔드와 같이 해결해야할 문제입니다."),
    NOT_FOUND(404, "C001", "존재하지 않는 url 입니다."),
    DUPLICATE_VALUE(409, "C001", "이미 존재하는 값입니다."),
    MEMBER_NOT_FOUND(404, "C001", "존재하지 않는 회원입니다."),
    ;

    // 에러 코드의 '코드 상태'을 반환한다.
    private final int status;

    // 에러 코드의 '코드간 구분 값'을 반환한다.
    private final String code;

    // 에러 코드의 '코드 메시지'을 반환한다.
    private final String message;

    RestErrorCode(final int status, final String code, final String message) {
        this.status = status;
        this.code = code;
        this.message = message;
    }
}
