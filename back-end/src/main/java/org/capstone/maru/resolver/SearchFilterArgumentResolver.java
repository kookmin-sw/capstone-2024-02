package org.capstone.maru.resolver;

import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.annotation.RequestQueryString;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.springframework.core.MethodParameter;
import org.springframework.web.bind.support.WebDataBinderFactory;
import org.springframework.web.context.request.NativeWebRequest;
import org.springframework.web.method.support.HandlerMethodArgumentResolver;
import org.springframework.web.method.support.ModelAndViewContainer;

@Slf4j
public class SearchFilterArgumentResolver implements HandlerMethodArgumentResolver {

    @Override
    public boolean supportsParameter(MethodParameter parameter) {
        return parameter.getParameterType().equals(SearchFilterRequest.class) &&
            parameter.hasParameterAnnotation(RequestQueryString.class);
    }

    @Override
    public Object resolveArgument(
        MethodParameter parameter,
        ModelAndViewContainer mavContainer,
        NativeWebRequest webRequest,
        WebDataBinderFactory binderFactory
    ) throws Exception {
        log.info("[Debug] resolveArgument called!");
        RequestQueryString annotation = parameter.getParameterAnnotation(
            RequestQueryString.class);
        assert annotation != null;

        String requestParameter = webRequest.getParameter(annotation.name());

        return SearchFilterRequest.fromJson(requestParameter);
    }
}
