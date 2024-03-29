package org.capstone.maru.dto;

import java.util.List;
import lombok.Builder;
import org.capstone.maru.domain.MemberCard;

@Builder
public record MemberCardDto(
    List<String> myFeatures
) {

    public static MemberCardDto from(MemberCard entity) {
        return MemberCardDto
            .builder()
            .myFeatures(entity.getMemberFeatures())
            .build();
    }
}
