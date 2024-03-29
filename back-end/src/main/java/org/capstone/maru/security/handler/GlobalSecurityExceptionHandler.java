package org.capstone.maru.security.handler;

import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.exception.RestErrorResponse;
import org.capstone.maru.security.exception.InvalidTokenException;
import org.capstone.maru.security.exception.MemberAccountExistentException;
import org.capstone.maru.security.exception.RefreshTokenNotFoundException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.AuthenticationException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@Slf4j
@RestControllerAdvice
public class GlobalSecurityExceptionHandler {

    @ExceptionHandler({AuthenticationException.class})
    @ResponseBody
    public ResponseEntity<RestErrorResponse> handleAuthenticationException(
        AuthenticationException ex
    ) {
        log.error("[Error] AuthenticationException occur!: {}", ex.getMessage());

        return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                             .body(RestErrorResponse.of(RestErrorCode.UNAUTHORIZED));
    }

    @ExceptionHandler({MemberAccountExistentException.class})
    @ResponseBody
    public ResponseEntity<RestErrorResponse> handleAlreadyHaveMemberAccountException(
        MemberAccountExistentException ex
    ) {
        log.error("[Error] MemberAccountExistentException occur!: {}", ex.getMessage());

        return ResponseEntity.status(ex.getErrorCode().getStatus())
                             .body(RestErrorResponse.of(ex.getErrorCode(), ex.getReason()));
    }

    @ExceptionHandler({InvalidTokenException.class})
    @ResponseBody
    public ResponseEntity<RestErrorResponse> handleInvalidTokenException(
        InvalidTokenException ex
    ) {
        log.error("[Error] InvalidTokenException occur!: {}", ex.getMessage());

        return ResponseEntity.status(ex.getErrorCode().getStatus())
                             .body(RestErrorResponse.of(ex.getErrorCode(), ex.getReason()));
    }

    @ExceptionHandler({RefreshTokenNotFoundException.class})
    @ResponseBody
    public ResponseEntity<RestErrorResponse> handleRefreshTokenNotFoundException(
        RefreshTokenNotFoundException ex
    ) {
        log.error("[Error] InvalidTokenException occur!: {}", ex.getMessage());

        return ResponseEntity.status(ex.getErrorCode().getStatus())
                             .body(RestErrorResponse.of(ex.getErrorCode(), ex.getReason()));
    }
}
