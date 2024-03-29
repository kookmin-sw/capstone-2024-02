package org.capstone.maru.domain.converter;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import org.capstone.maru.domain.constant.RentalType;

@Converter
public class RentalTypeConverter implements AttributeConverter<RentalType, String> {

    @Override
    public String convertToDatabaseColumn(RentalType rentalType) {
        return rentalType.getDescription();
    }

    @Override
    public RentalType convertToEntityAttribute(String description) {
        return RentalType.of(description);
    }
}
