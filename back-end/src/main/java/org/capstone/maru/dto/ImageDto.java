package org.capstone.maru.dto;

import lombok.Builder;

@Builder
public record ImageDto(
    String url,
    String fileName
) {

    public static ImageDto from(String url, String fileName) {
        return ImageDto.builder()
            .url(url)
            .fileName(fileName)
            .build();
    }
}
