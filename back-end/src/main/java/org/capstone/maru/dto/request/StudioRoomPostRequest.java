package org.capstone.maru.dto.request;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.util.List;
import org.capstone.maru.annotation.ImageFilesConstraints;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.Address.CITY;
import org.capstone.maru.domain.constant.FloorType;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.domain.jsonb.ExtraOption;
import org.capstone.maru.dto.MemberCardDto;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.RoomInfoDto;
import org.capstone.maru.dto.StudioRoomPostDto;
import org.hibernate.validator.constraints.Length;

public record StudioRoomPostRequest(
    @ImageFilesConstraints
    List<@Valid ImageFileData> imageFilesData,
    @Valid
    PostData postData,
    @Valid
    TransactionData transactionData,
    @Valid
    RoomDetailData roomDetailData,
    @Valid
    LocationData locationData,
    MemberFeatureRequest roomMateCardData,
    @NotNull(message = "참가자를 반드시 넣어주세요.")
    List<String> participationMemberIds
) {

    // -- 생성자 -- //
    public StudioRoomPostDto toBaseStudioRoomPostDto(String publisherGender) {
        return StudioRoomPostDto
            .builder()
            .title(postData.title)
            .content(postData.content)
            .publisherGender(publisherGender)
            .build();
    }

    public MemberCardDto toMemberCardDto() {
        return MemberCardDto
            .builder()
            .location(roomMateCardData.location())
            .myFeatures(roomMateCardData.features())
            .build();
    }

    public List<RoomImageDto> toRoomImagesDto() {
        return imageFilesData
            .stream()
            .map(imageFileData ->
                RoomImageDto
                    .builder()
                    .fileName(imageFileData.fileName)
                    .isThumbnail(imageFileData.isThumbNail)
                    .order(imageFileData.order)
                    .build()
            )
            .toList();
    }

    public RoomInfoDto toRoomInfoDto() {
        return RoomInfoDto
            .builder()
            .address(
                Address.of(locationData.city, locationData.oldAddress,
                    locationData.roadAddress, locationData.detailAddress
                )
            )
            .roomType(roomDetailData.roomType)
            .floorType(roomDetailData.floorType)
            .size(roomDetailData.size)
            .numberOfRoom(roomDetailData.numberOfRoom)
            .numberOfBathRoom(roomDetailData.numberOfBathRoom)
            .hasLivingRoom(roomDetailData.hasLivingRoom)
            .rentalType(transactionData.rentalType)
            .expectedPayment(transactionData.expectedPayment)
            .recruitmentCapacity(roomDetailData.recruitmentCapacity)
            .extraOption(roomDetailData.extraOption)
            .build();
    }

    // -- Nested -- //
    public record ImageFileData(
        @NotBlank(message = "업로드 된 이미지 파일 이름을 작성해야 합니다.")
        String fileName,
        @NotNull(message = "썸네일인지 아닌지 boolean 값으로 작성해야 합니다.")
        Boolean isThumbNail,
        @Min(value = 1, message = "순서는 1부터 시작합니다.")
        @Max(value = 10, message = "11개 이상의 순서는 있을 수 없습니다.")
        Short order
    ) {

    }

    public record PostData(
        @NotBlank(message = "제목을 작성해야 합니다.")
        @Length(max = 50, message = "제목은 50자를 넘어갈 수 없습니다.")
        String title,
        @NotBlank(message = "상세 정보를 작성해야 합니다.")
        String content
    ) {

    }

    public record TransactionData(
        @NotNull(message = "거래 방식 입력 값이 잘못 되었습니다.")
        RentalType rentalType,
        @Min(value = 0, message = "희망 금액은 음수 일 수 없습니다.")
        @Max(value = 10_0000_0000, message = "희망 금액에 이상치 값이 입력 되었습니다.")
        Long expectedPayment
    ) {

    }

    public record RoomDetailData(
        @NotNull(message = "방 종류 입력 값이 잘못 되었습니다.")
        RoomType roomType,
        @NotNull(message = "층 종류 입력 값이 잘못 되었습니다.")
        FloorType floorType,
        @Min(value = 0, message = "방 크기는 음수 일 수 없습니다.")
        @Max(value = 100, message = "방 크기에 이상치 값이 입력 되었습니다.")
        Short size,
        @Min(value = 0, message = "방 개수는 음수 일 수 없습니다.")
        @Max(value = 10, message = "방 개수에 이상치 값이 입력 되었습니다.")
        Short numberOfRoom,
        @Min(value = 0, message = "화장실 개수는 음수 일 수 없습니다.")
        @Max(value = 10, message = "화장실 개수에 이상치 값이 입력 되었습니다.")
        Short numberOfBathRoom,
        @NotNull(message = "거실 유무 입력 값이 잘못 되었습니다.")
        Boolean hasLivingRoom,
        @Min(value = 0, message = "모집 인원은 음수 일 수 없습니다.")
        @Max(value = 10, message = "모집 인원에 이상치 값이 입력 되었습니다.")
        Short recruitmentCapacity,
        ExtraOption extraOption
    ) {

    }

    public record LocationData(
        @NotNull(message = "도시 입력 값이 잘못 되었습니다.")
        CITY city,
        String oldAddress,
        String roadAddress,
        String detailAddress
    ) {

    }

    // -- 편의 메서드 -- //
}
