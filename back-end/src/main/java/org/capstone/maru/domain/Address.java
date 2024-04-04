package org.capstone.maru.domain;

import com.google.gson.annotations.SerializedName;
import jakarta.persistence.Column;
import jakarta.persistence.Convert;
import jakarta.persistence.Embeddable;
import java.util.Arrays;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.capstone.maru.domain.converter.AddressCityConverter;

@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Embeddable
@Getter
public class Address {

    @Convert(converter = AddressCityConverter.class)
    @Column(nullable = false)
    private CITY city;

    @Column(nullable = false)
    private String oldAddress; // 지번 주소

    @Column(nullable = false)
    private String roadAddress; // 도로명 주소

    private String detailAddress; // 상세 주소

    private String stationName; // 주변 지하철역

    private Short stationTime; // 지하철역까지 걸리는 시간

    private Short busStopTime; // 버스 정류장까지 걸리는 시간

    private String schoolName; // 학교

    private Short schoolTime; // 학교까지 걸리는 시간

    private Short convenienceStoreTime; // 편의점까지 걸리는 시간

    // -- 생성 메서드 -- //
    private Address(CITY city, String oldAddress, String roadAddress, String detailAddress,
        String stationName, Short stationTime, Short busStopTime, String schoolName,
        Short schoolTime,
        Short convenienceStoreTime) {
        this.city = city;
        this.oldAddress = oldAddress;
        this.roadAddress = roadAddress;
        this.detailAddress = detailAddress;
        this.stationName = stationName;
        this.stationTime = stationTime;
        this.busStopTime = busStopTime;
        this.schoolName = schoolName;
        this.schoolTime = schoolTime;
        this.convenienceStoreTime = convenienceStoreTime;
    }

    public static Address of(
        CITY city,
        String oldAddress,
        String roadAddress,
        String detailAddress,
        String stationName,
        Short stationTime,
        Short busStopTime,
        String schoolName,
        Short schoolTime,
        Short convenienceStoreTime
    ) {
        return new Address(
            city,
            oldAddress,
            roadAddress,
            detailAddress,
            stationName,
            stationTime,
            busStopTime,
            schoolName,
            schoolTime,
            convenienceStoreTime
        );
    }

    @Getter
    public enum CITY {
        @SerializedName("SEOUL")
        SEOUL("서울특별시"),
        @SerializedName("BUSAN")
        BUSAN("부산광역시");

        private final String name;

        CITY(String name) {
            this.name = name;
        }

        public static CITY of(String name) {
            return Arrays.stream(CITY.values())
                         .filter(v -> v.getName().equals(name))
                         .findAny()
                         .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 도시명입니다."));
        }
    }
}
