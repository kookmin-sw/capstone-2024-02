package org.capstone.maru.exception;

import lombok.Builder;

/**
 * @param status   에러 상태 코드
 * @param code     에러 구분 코드
 * @param errorMsg 에러 메시지
 * @param reason   에러 이유
 */
@Builder
public record RestErrorResponse(
    int status,
    String code,
    String errorMsg,
    String reason
) {

    /**
     * Global Exception 전송 타입
     *
     * @param code ErrorCode
     * @return ErrorResponse
     */
    public static RestErrorResponse of(final RestErrorCode code) {
        return RestErrorResponse.builder()
                                .status(code.getStatus())
                                .code(code.getCode())
                                .errorMsg(code.getMessage())
                                .build();
    }

    /**
     * Global Exception 전송 타입
     *
     * @param code   ErrorCode
     * @param reason String
     * @return ErrorResponse
     */
    public static RestErrorResponse of(final RestErrorCode code, final String reason) {
        return RestErrorResponse.builder()
                                .status(code.getStatus())
                                .code(code.getCode())
                                .errorMsg(code.getMessage())
                                .reason(reason)
                                .build();
    }

}
