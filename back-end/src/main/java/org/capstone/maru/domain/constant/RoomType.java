package org.capstone.maru.domain.constant;

import com.google.gson.annotations.SerializedName;
import java.util.Arrays;
import lombok.Getter;

@Getter
public enum RoomType {
    @SerializedName("0")
    ONE_ROOM("원룸", 0),
    @SerializedName("1")
    TWO_ROOM_VILLA("빌라/투룸이상", 1),
    @SerializedName("2")
    APT("아파트", 2),
    @SerializedName("3")
    OFFICE_TEL("원룸/오피스텔", 3);


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
