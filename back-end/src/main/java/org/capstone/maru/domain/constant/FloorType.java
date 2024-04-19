package org.capstone.maru.domain.constant;

import com.google.gson.annotations.SerializedName;
import java.util.Arrays;
import lombok.Getter;

@Getter
public enum FloorType {
    @SerializedName("0")
    GROUND("지상", 0),
    @SerializedName("1")
    SEMI_BASEMENT("반지하", 1),
    @SerializedName("2")
    PENTHOUSE("옥탑", 2);


    private final String description;
    private final int code;

    FloorType(String description, int code) {
        this.description = description;
        this.code = code;
    }

    public static FloorType of(int code) {
        return Arrays.stream(FloorType.values())
                     .filter(v -> v.getCode() == code)
                     .findAny()
                     .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 층 유형입니다."));
    }
}
