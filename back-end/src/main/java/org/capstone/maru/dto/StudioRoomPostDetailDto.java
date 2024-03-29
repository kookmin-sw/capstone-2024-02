package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.Random;
import java.util.Set;
import lombok.Builder;
import org.capstone.maru.domain.StudioRoomPost;

@Builder
public record StudioRoomPostDetailDto(
    Long id,
    String title,
    String content,
    String publisherGender,
    Set<RoomImageDto> roomImages,
    MemberAccountWithCardsDto publisherAccount,
    RoomInfoDto roomInfo,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static StudioRoomPostDetailDto from(StudioRoomPost entity) {
        return StudioRoomPostDetailDto
            .builder()
            .id(entity.getId())
            .title(entity.getTitle())
            .content(entity.getContent())
            .roomImages(createDummyRoomImagesDto())
            .publisherGender(entity.getPublisherGender())
            .publisherAccount(MemberAccountWithCardsDto.from(entity.getPublisherAccount()))
            .roomInfo(RoomInfoDto.from(entity.getRoomInfo()))
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .build();
    }

    /**
     * 이미지 기능 구현 전이라 임시 이미지 데이터 작성. 코드 이미지 기능 구현 후 코드 지워주기
     */
    private static Set<RoomImageDto> createDummyRoomImagesDto() {
        return Set.of(
            createDummyRoomImageDto(true),
            createDummyRoomImageDto(false),
            createDummyRoomImageDto(false)
        );
    }

    private static RoomImageDto createDummyRoomImageDto(Boolean isThumbnail) {
        final String imageUrl = "http://mstatic1.e-himart.co.kr/contents/content/upload/style/20200914/950958/thumbnail_750_propse_tagging_4920.jpg";

        return RoomImageDto
            .builder()
            .id(new Random().nextLong())
            .fileName("dummy room image")
            .storeImagePath(imageUrl)
            .isThumbnail(isThumbnail)
            .createdAt(LocalDateTime.now())
            .createdBy("tester")
            .modifiedAt(LocalDateTime.now())
            .createdBy("tester")
            .build();
    }
}