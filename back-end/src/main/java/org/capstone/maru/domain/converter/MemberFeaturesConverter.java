package org.capstone.maru.domain.converter;

import com.google.gson.Gson;
import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import java.util.List;

@Converter
public class MemberFeaturesConverter implements AttributeConverter<List<String>, String> {

    @Override
    public String convertToDatabaseColumn(List strings) {
        return new Gson().toJson(strings);
    }

    @Override
    public List<String> convertToEntityAttribute(String s) {
        return new Gson().fromJson(s, List.class);
    }
}
