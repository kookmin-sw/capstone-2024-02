package org.capstone.maru.dto.response;

import java.io.Serializable;
import java.time.LocalDateTime;
import lombok.Builder;
import org.capstone.maru.dto.RoomImageDto;

@Builder
public record RoomImageResponse(
    Long id,
    String storeImagePath
) {

    public static RoomImageResponse from(RoomImageDto dto) {
        return RoomImageResponse
            .builder()
            .id(dto.id())
            .storeImagePath(dto.storeImagePath())
            .build();
    }
}