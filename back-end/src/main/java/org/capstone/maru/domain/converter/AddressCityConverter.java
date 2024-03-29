package org.capstone.maru.domain.converter;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.Address.CITY;

@Converter
public class AddressCityConverter implements AttributeConverter<Address.CITY, String> {

    @Override
    public String convertToDatabaseColumn(CITY city) {
        return city.getName();
    }

    @Override
    public CITY convertToEntityAttribute(String cityName) {
        return CITY.of(cityName);
    }
}
