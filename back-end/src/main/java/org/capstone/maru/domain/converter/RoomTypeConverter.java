package org.capstone.maru.domain.converter;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import org.capstone.maru.domain.constant.RoomType;

@Converter
public class RoomTypeConverter implements AttributeConverter<RoomType, String> {


    @Override
    public String convertToDatabaseColumn(RoomType roomType) {
        return roomType.getDescription();
    }

    @Override
    public RoomType convertToEntityAttribute(String description) {
        return RoomType.of(description);
    }
}
