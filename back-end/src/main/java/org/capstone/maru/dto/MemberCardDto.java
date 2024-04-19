package org.capstone.maru.dto;

import java.util.List;
import lombok.Builder;
import org.capstone.maru.domain.FeatureCard;

@Builder
public record MemberCardDto(
    Long id,
    String location,
    List<String> myFeatures
) {

    public static MemberCardDto from(FeatureCard entity) {
        return MemberCardDto
            .builder()
            .id(entity.getId())
            .location(entity.getLocation())
            .myFeatures(entity.getMemberFeatures())
            .build();
    }

    public FeatureCard toEntity() {
        return FeatureCard.of(
            this.myFeatures
        );
    }
}
