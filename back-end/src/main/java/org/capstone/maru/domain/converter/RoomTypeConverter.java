package org.capstone.maru.domain.converter;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import org.capstone.maru.domain.constant.RoomType;

@Converter
public class RoomTypeConverter implements AttributeConverter<RoomType, Integer> {


    @Override
    public Integer convertToDatabaseColumn(RoomType roomType) {
        return roomType.getCode();
    }

    @Override
    public RoomType convertToEntityAttribute(Integer code) {
        return RoomType.of(code);
    }
}
