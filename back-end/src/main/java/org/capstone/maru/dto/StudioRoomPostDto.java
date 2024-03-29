package org.capstone.maru.dto;

import java.time.LocalDateTime;
import java.util.Random;
import java.util.Set;

import lombok.Builder;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.RoomInfo;
import org.capstone.maru.domain.StudioRoomPost;

@Builder
public record StudioRoomPostDto(
    Long id,
    String title,
    String content,
    String publisherGender,
    Set<RoomImageDto> roomImages,
    MemberAccountDto publisherAccount,
    RoomInfoDto roomInfo,
    LocalDateTime createdAt,
    String createdBy,
    LocalDateTime modifiedAt,
    String modifiedBy
) {

    public static StudioRoomPostDto from(StudioRoomPost entity) {
        return StudioRoomPostDto
            .builder()
            .id(entity.getId())
            .title(entity.getTitle())
            .content(entity.getContent())
            .roomImages(createDummyRoomImagesDto())
            .publisherGender(entity.getPublisherGender())
            .publisherAccount(MemberAccountDto.from(entity.getPublisherAccount()))
            .roomInfo(RoomInfoDto.from(entity.getRoomInfo()))
            .createdAt(entity.getCreatedAt())
            .createdBy(entity.getCreatedBy())
            .modifiedAt(entity.getModifiedAt())
            .modifiedBy(entity.getModifiedBy())
            .build();
    }

    public StudioRoomPost toEntity(MemberAccount publisherAccountEntity, RoomInfo roomInfoEntity) {
        return StudioRoomPost.of(
            title,
            content,
            publisherGender,
            publisherAccountEntity,
            roomInfoEntity
        );
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

    public RoomImageDto thumbnail() {
        return roomImages
            .stream()
            .filter(RoomImageDto::isThumbnail)
            .findAny()
            .orElseThrow(IllegalArgumentException::new);
    }
}