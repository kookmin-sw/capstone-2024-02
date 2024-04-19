package org.capstone.maru.domain.converter;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import org.capstone.maru.domain.constant.FloorType;

@Converter
public class FloorTypeConverter implements AttributeConverter<FloorType, Integer> {

    @Override
    public Integer convertToDatabaseColumn(FloorType floorType) {
        return floorType.getCode();
    }

    @Override
    public FloorType convertToEntityAttribute(Integer code) {
        return FloorType.of(code);
    }
}
