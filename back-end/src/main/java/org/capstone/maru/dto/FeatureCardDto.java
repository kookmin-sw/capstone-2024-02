package org.capstone.maru.dto;

import lombok.Builder;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.domain.jsonb.MemberFeatures;

@Builder
public record FeatureCardDto(
    Long id,
    String location,
    MemberFeatures myFeatures
) {

    public static FeatureCardDto from(FeatureCard entity) {
        return FeatureCardDto
            .builder()
            .id(entity.getId())
            .location(entity.getLocation())
            .myFeatures(entity.getMemberFeatures())
            .build();
    }

    public FeatureCard toEntity() {
        return FeatureCard.of(
            this.location,
            this.myFeatures
        );
    }
}
