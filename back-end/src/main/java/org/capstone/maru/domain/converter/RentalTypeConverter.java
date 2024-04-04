package org.capstone.maru.domain.converter;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import org.capstone.maru.domain.constant.RentalType;

@Converter
public class RentalTypeConverter implements AttributeConverter<RentalType, Integer> {

    @Override
    public Integer convertToDatabaseColumn(RentalType rentalType) {
        return rentalType.getCode();
    }

    @Override
    public RentalType convertToEntityAttribute(Integer code) {
        return RentalType.of(code);
    }
}
