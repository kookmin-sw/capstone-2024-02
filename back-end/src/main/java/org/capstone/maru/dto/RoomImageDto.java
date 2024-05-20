package org.capstone.maru.dto;

import java.time.LocalDateTime;
import lombok.Builder;
import org.capstone.maru.domain.RoomImage;
import org.capstone.maru.domain.SharedRoomPost;
import org.capstone.maru.domain.StudioRoomPost;

@Builder
public record RoomImageDto(
    String fileName,
    Boolean isThumbnail,
    Short order,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static RoomImageDto from(RoomImage entity) {
        return RoomImageDto
            .builder()
            .fileName(entity.getFileName())
            .isThumbnail(entity.getIsThumbnail())
            .order(entity.getOrderNumber())
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .build();
    }

    public RoomImage toEntity(SharedRoomPost roomPost) {
        return RoomImage.of(fileName, isThumbnail, order, roomPost);
    }

}