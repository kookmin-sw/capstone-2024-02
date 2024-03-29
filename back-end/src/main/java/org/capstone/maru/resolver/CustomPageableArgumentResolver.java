package org.capstone.maru.resolver;

import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.exception.PageSizeOutOfBoundsException;
import org.capstone.maru.exception.RestErrorCode;
import org.springframework.core.MethodParameter;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableHandlerMethodArgumentResolver;
import org.springframework.web.bind.support.WebDataBinderFactory;
import org.springframework.web.context.request.NativeWebRequest;
import org.springframework.web.method.support.ModelAndViewContainer;

@Slf4j
public class CustomPageableArgumentResolver extends PageableHandlerMethodArgumentResolver {

    @Override
    public Pageable resolveArgument(
        final MethodParameter methodParameter,
        final ModelAndViewContainer mavContainer,
        final NativeWebRequest webRequest,
        final WebDataBinderFactory binderFactory
    ) {
        String pageSize = webRequest
            .getParameter(getParameterNameToUse(getSizeParameterName(), methodParameter));

        validate(pageSize);

        return super.resolveArgument(methodParameter, mavContainer, webRequest, binderFactory);
    }

    private void validate(final String pageSize) {
        if (pageSize == null) {
            return;
        }

        if (Integer.parseInt(pageSize) > 100) {
            throw new PageSizeOutOfBoundsException(RestErrorCode.PAGE_SIZE_OUT_OF_BOUNDS,
                "페이지 사이즈가 100 이상 일 수 없습니다.");
        }

        if (Integer.parseInt(pageSize) < 0) {
            throw new PageSizeOutOfBoundsException(RestErrorCode.PAGE_SIZE_OUT_OF_BOUNDS,
                "페이지 사이즈가 음수 일 수 없습니다.");
        }
    }
}
