package org.capstone.maru.dto;

import java.time.LocalDateTime;
import lombok.Builder;
import org.capstone.maru.domain.SharedRoomPost;
import org.capstone.maru.domain.StudioRoomPost;

@Builder
public record SharedRoomPostDto(
    Long id,
    String title,
    String type,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static SharedRoomPostDto from(SharedRoomPost entity) {
        return SharedRoomPostDto
            .builder()
            .id(entity.getId())
            .title(entity.getTitle())
            .type(entity instanceof StudioRoomPost ? "studio" : "dormitory")
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .build();
    }
}
