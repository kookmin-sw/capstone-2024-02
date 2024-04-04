package org.capstone.maru.validator;

import jakarta.validation.ConstraintValidator;
import jakarta.validation.ConstraintValidatorContext;
import java.util.List;
import org.capstone.maru.annotation.ImageFilesConstraints;
import org.capstone.maru.dto.request.StudioRoomPostRequest.ImageFilesData.ImageFileData;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

public class ImageFilesConstraintsValidator implements
    ConstraintValidator<ImageFilesConstraints, List<ImageFileData>> {

    @Override
    public boolean isValid(List<ImageFileData> value, ConstraintValidatorContext context) {
        return (!value.isEmpty() && value.size() <= 10) &&
            value.stream()
                 .anyMatch(ImageFileData::isThumbNail);
    }
}
