package org.capstone.maru.dto;

import java.time.LocalDateTime;
import lombok.Builder;
import org.capstone.maru.domain.RoomImage;
import org.capstone.maru.domain.StudioRoomPost;

@Builder
public record RoomImageDto(
    Long id,
    String fileName,
    String storeImagePath,
    Boolean isThumbnail,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static RoomImageDto from(RoomImage entity) {
        return RoomImageDto
            .builder()
            .id(entity.getId())
            .fileName(entity.getFileName())
            .storeImagePath(entity.getStoreImagePath())
            .isThumbnail(entity.getIsThumbnail())
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .build();
    }

    public RoomImage toEntity(StudioRoomPost studioRoomPost) {
        return RoomImage.of(fileName, storeImagePath, isThumbnail, studioRoomPost);
    }

}