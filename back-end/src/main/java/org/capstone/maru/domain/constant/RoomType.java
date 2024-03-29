package org.capstone.maru.domain.constant;

import com.google.gson.annotations.SerializedName;
import java.util.Arrays;
import lombok.Getter;

@Getter
public enum RoomType {
    @SerializedName("0")
    VILLA("주택/빌라 (원투쓰리룸)"),
    @SerializedName("1")
    OFFICE_TEL("오피스텔"),
    @SerializedName("2")
    APT("아파트");

    private final String description;

    RoomType(String description) {
        this.description = description;
    }

    public static RoomType of(String description) {
        return Arrays.stream(RoomType.values())
                     .filter(v -> v.getDescription().equals(description))
                     .findAny()
                     .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 거래 유형입니다."));
    }
}
