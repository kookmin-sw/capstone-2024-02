package org.capstone.maru.exception;

import com.google.gson.stream.MalformedJsonException;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.support.DefaultMessageSourceResolvable;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.web.bind.MethodArgumentNotValidException;
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

        return ResponseEntity
            .status(HttpStatus.NOT_FOUND)
            .body(RestErrorResponse.of(RestErrorCode.NOT_FOUND));
    }

    @ExceptionHandler({PageSizeOutOfBoundsException.class})
    public ResponseEntity<RestErrorResponse> handlePageSizeOutOfBoundsException(
        PageSizeOutOfBoundsException ex
    ) {
        return ResponseEntity
            .status(ex.getErrorCode().getStatus())
            .body(RestErrorResponse.of(ex.getErrorCode(), ex.getReason()));
    }

    @ExceptionHandler({MethodArgumentNotValidException.class})
    public ResponseEntity<RestErrorResponse> handleValidationExceptions(
        MethodArgumentNotValidException ex) {
        List<String> errorMessages = ex.getBindingResult().getAllErrors().stream()
                                       .map(DefaultMessageSourceResolvable::getDefaultMessage)
                                       .toList();

        return ResponseEntity
            .status(HttpStatus.BAD_REQUEST)
            .body(RestErrorResponse.of(HttpStatus.BAD_REQUEST, errorMessages));
    }

    @ExceptionHandler(MalformedJsonException.class)
    public ResponseEntity<RestErrorResponse> handleMalformedJSONException(
        MalformedJsonException ex
    ) {
        return ResponseEntity
            .status(HttpStatus.BAD_REQUEST)
            .body(RestErrorResponse.of(HttpStatus.BAD_REQUEST,
                "사용자 입력 form data가 잘못 되었습니다."));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<RestErrorResponse> handleIllegalArgumentException(
        IllegalArgumentException ex
    ) {
        return ResponseEntity
            .status(HttpStatus.BAD_REQUEST)
            .body(RestErrorResponse.of(HttpStatus.BAD_REQUEST, ex.getMessage()));
    }

    @ExceptionHandler(PostNotFoundException.class)
    public ResponseEntity<RestErrorResponse> handlePostNotFoundException(
        PostNotFoundException ex
    ) {
        return ResponseEntity
            .status(ex.getErrorCode().getStatus())
            .body(RestErrorResponse.of(ex.getErrorCode(), ex.getReason()));
    }
}
