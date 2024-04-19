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

    // -- 생성 메서드 -- //
    private Address(CITY city, String oldAddress, String roadAddress, String detailAddress) {
        this.city = city;
        this.oldAddress = oldAddress;
        this.roadAddress = roadAddress;
        this.detailAddress = detailAddress;
    }

    public static Address of(
        CITY city,
        String oldAddress,
        String roadAddress,
        String detailAddress
    ) {
        return new Address(
            city,
            oldAddress,
            roadAddress,
            detailAddress
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
