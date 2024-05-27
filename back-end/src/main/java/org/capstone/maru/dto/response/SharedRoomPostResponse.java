package org.capstone.maru.dto.response;

import java.time.LocalDateTime;
import lombok.Builder;
import org.capstone.maru.dto.SharedRoomPostDto;

@Builder
public record SharedRoomPostResponse(
    Long id,
    String title,
    String type,
    LocalDateTime createdAt,
    LocalDateTime modifiedAt
) {

    public static SharedRoomPostResponse from(SharedRoomPostDto dto) {
        return SharedRoomPostResponse
            .builder()
            .id(dto.id())
            .title(dto.title())
            .type(dto.type())
            .createdAt(dto.createdAt())
            .modifiedAt(dto.modifiedAt())
            .build();
    }

}
