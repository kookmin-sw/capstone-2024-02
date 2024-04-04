package org.capstone.maru.dto.response;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import lombok.Builder;
import org.capstone.maru.dto.RoomImageDto;

@Builder
public record RoomImageResponse(
    String fileName,
    Boolean isThumbnail,
    Short order
) {

    public static RoomImageResponse from(RoomImageDto dto) {
        return RoomImageResponse
            .builder()
            .fileName(convertFileName(dto.fileName()))
            .isThumbnail(dto.isThumbnail())
            .order(dto.order())
            .build();
    }

    // 테스트 더미 데이터만을 위한 코드 배포 시 수정해 줄 것
    private static String convertFileName(String fileName) {
        Pattern pattern = Pattern.compile("(.+\\.jpg)(\\d+)");
        Matcher matcher = pattern.matcher(fileName);

        if (matcher.find()) {
            return matcher.group(1);
        }

        return fileName;
    }
}