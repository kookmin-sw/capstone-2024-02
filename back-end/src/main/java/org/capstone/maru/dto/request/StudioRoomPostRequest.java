package org.capstone.maru.dto.request;

import com.google.gson.Gson;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.Address.CITY;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.dto.MemberAccountDto;
import org.capstone.maru.dto.MemberAccountWithCardsDto;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.RoomInfoDto;
import org.capstone.maru.dto.StudioRoomPostDetailDto;
import org.capstone.maru.dto.StudioRoomPostDto;
import org.springframework.web.multipart.MultipartFile;

public record StudioRoomPostRequest(
    List<MultipartFile> files,
    String postData,
    String transactionData,
    String roomDetailData,
    String locationData,
    PostData postInfoData,
    TransactionData transactionInfoData,
    RoomDetailData roomInfoData,
    LocationData locationInfoData
) {

    // -- 생성자 -- //
    public StudioRoomPostRequest(List<MultipartFile> files, String postData, String transactionData,
        String roomDetailData, String locationData, PostData postInfoData,
        TransactionData transactionInfoData, RoomDetailData roomInfoData,
        LocationData locationInfoData) {
        this.files = files;
        this.postData = postData;
        this.transactionData = transactionData;
        this.roomDetailData = roomDetailData;
        this.locationData = locationData;

        this.postInfoData = PostData.fromJson(postData);
        this.transactionInfoData = TransactionData.fromJson(transactionData);
        this.roomInfoData = RoomDetailData.fromJson(roomDetailData);
        this.locationInfoData = LocationData.fromJson(locationData);
    }

    public StudioRoomPostDto toBaseStudioRoomPostDto() {
        return StudioRoomPostDto
            .builder()
            .title(postInfoData.title)
            .content(postInfoData.content)
            .build();
    }

    public Set<RoomImageDto> toRoomImagesDto() {
        return createDummyRoomImagesDto();
    }

    public RoomInfoDto toRoomInfoDto() {
        return RoomInfoDto
            .builder()
            .address(
                Address.of(locationInfoData.city, locationInfoData.oldAddress,
                    locationInfoData.roadAddress, locationInfoData.detailAddress,
                    locationInfoData.stationName,
                    locationInfoData.stationTime, locationInfoData.busStopTime,
                    locationInfoData.schoolName, locationInfoData.schoolTime,
                    locationInfoData.convenienceStoreTime
                )
            )
            .roomType(roomInfoData.roomType)
            .size(roomInfoData.size)
            .numberOfRoom(roomInfoData.numberOfRoom)
            .rentalType(transactionInfoData.rentalType)
            .price(transactionInfoData.price)
            .monthlyFee(transactionInfoData.monthlyFee)
            .managementFee(transactionInfoData.managementFee)
            .recruitmentCapacity(roomInfoData.recruitmentCapacity)
            .build();
    }

    // -- Nested -- //
    public record PostData(
        String title,
        String content
    ) {

        private static PostData fromJson(String json) {
            Gson gson = new Gson();
            return gson.fromJson(json, PostData.class);
        }
    }

    public record TransactionData(
        RentalType rentalType,
        Long price,
        Long monthlyFee,
        Long managementFee
    ) {

        private static TransactionData fromJson(String json) {
            Gson gson = new Gson();
            return gson.fromJson(json, TransactionData.class);
        }

    }

    public record RoomDetailData(
        RoomType roomType,
        Short size,
        Short numberOfRoom,
        Short recruitmentCapacity
    ) {

        private static RoomDetailData fromJson(String json) {
            Gson gson = new Gson();
            return gson.fromJson(json, RoomDetailData.class);
        }

    }

    public record LocationData(
        CITY city,
        String oldAddress,
        String roadAddress,
        String detailAddress,
        String stationName,
        Short stationTime,
        Short busStopTime,
        String schoolName,
        String schoolTime,
        String convenienceStoreTime
    ) {

        private static LocationData fromJson(String json) {
            Gson gson = new Gson();
            return gson.fromJson(json, LocationData.class);
        }

    }

    // -- 편의 메서드 -- //

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
