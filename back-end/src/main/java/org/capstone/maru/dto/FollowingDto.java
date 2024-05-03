package org.capstone.maru.dto;

import java.util.List;
import java.util.Map;
import lombok.Builder;

@Builder
public record FollowingDto(
    Map<String, List<String>> followingList
) {

    public static FollowingDto from(Map<String, List<String>> followingList) {
        return FollowingDto
            .builder()
            .followingList(followingList)
            .build();
    }
}
