package org.capstone.maru.annotation;

import static java.lang.annotation.ElementType.ANNOTATION_TYPE;
import static java.lang.annotation.ElementType.CONSTRUCTOR;
import static java.lang.annotation.ElementType.FIELD;
import static java.lang.annotation.ElementType.METHOD;
import static java.lang.annotation.ElementType.PARAMETER;
import static java.lang.annotation.ElementType.TYPE_USE;
import static java.lang.annotation.RetentionPolicy.RUNTIME;

import jakarta.validation.Constraint;
import jakarta.validation.Payload;
import java.lang.annotation.Documented;
import java.lang.annotation.Retention;
import java.lang.annotation.Target;
import org.capstone.maru.validator.ImageFilesConstraintsValidator;

@Target({METHOD, FIELD, ANNOTATION_TYPE, CONSTRUCTOR, PARAMETER, TYPE_USE})
@Retention(RUNTIME)
@Documented
@Constraint(validatedBy = ImageFilesConstraintsValidator.class)
public @interface ImageFilesConstraints {

    String message() default "게시글 작성 시 썸네일 이미지는 반드시 존재 해야하며 10개 이하의 이미지만 등록 가능합니다.";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
