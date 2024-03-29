package org.capstone.maru.exception;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.servlet.resource.NoResourceFoundException;

@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler({NoResourceFoundException.class})
    public ResponseEntity<RestErrorResponse> handleNoResourceFoundException(
        NoResourceFoundException ex
    ) {
        log.error("[Error] NoResourceFoundException occur!: {}", ex.getMessage());

        return ResponseEntity.status(HttpStatus.NOT_FOUND)
                             .body(RestErrorResponse.of(RestErrorCode.NOT_FOUND));
    }

    @ExceptionHandler({PageSizeOutOfBoundsException.class})
    public ResponseEntity<RestErrorResponse> handlePageSizeOutOfBoundsException(
        PageSizeOutOfBoundsException ex
    ) {
        return ResponseEntity.status(ex.getErrorCode().getStatus())
                             .body(RestErrorResponse.of(ex.getErrorCode(), ex.getReason()));
    }
}
