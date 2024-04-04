package org.capstone.maru.domain.constant;

import com.google.gson.annotations.SerializedName;
import java.util.Arrays;
import lombok.Getter;

@Getter
public enum RentalType {
    @SerializedName("0")
    MONTHLY("월세", 0),
    @SerializedName("1")
    JEONSE("전세", 1);

    private final String description;
    private final int code;

    RentalType(String description, int code) {
        this.description = description;
        this.code = code;
    }

    public static RentalType of(int code) {
        return Arrays.stream(RentalType.values())
                     .filter(v -> v.getCode() == code)
                     .findAny()
                     .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 거래 유형입니다."));
    }
}
