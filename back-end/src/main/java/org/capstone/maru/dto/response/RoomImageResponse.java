package org.capstone.maru.dto.response;

import lombok.Builder;
import org.capstone.maru.dto.RoomImageDto;

@Builder
public record RoomImageResponse(
    String fileName,
    Boolean isThumbnail,
    Short order
) {

    public static RoomImageResponse from(RoomImageDto dto) {
        return RoomImageResponse
            .builder()
            .fileName(dto.fileName())
            .isThumbnail(dto.isThumbnail())
            .order(dto.order())
            .build();
    }
}