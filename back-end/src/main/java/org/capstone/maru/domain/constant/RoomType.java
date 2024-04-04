package org.capstone.maru.domain.constant;

import com.google.gson.annotations.SerializedName;
import java.util.Arrays;
import lombok.Getter;

@Getter
public enum RoomType {
    @SerializedName("0")
    ONE_ROOM_VILLA("원룸/빌라", 0),
    @SerializedName("1")
    TWO_ROOM_VILLA("투룸/빌라", 1),
    @SerializedName("2")
    THREE_ROOM_VILLA("쓰리룸이상/빌라", 2),

    @SerializedName("3")
    OFFICE_TEL("원룸/오피스텔", 3),
    @SerializedName("4")
    TWO_ROOM_OFFICE_TEL("투룸/오피스텔", 4),
    @SerializedName("5")
    THREE_ROOM_OFFICE_TEL("쓰리룸이상/오피스텔", 5),

    @SerializedName("6")
    APT("아파트", 6);


    private final String description;
    private final int code;

    RoomType(String description, int code) {
        this.description = description;
        this.code = code;
    }

    public static RoomType of(int code) {
        return Arrays.stream(RoomType.values())
                     .filter(v -> v.getCode() == code)
                     .findAny()
                     .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 거래 유형입니다."));
    }
}
