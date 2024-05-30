package org.capstone.maru.service;

import com.amazonaws.HttpMethod;
import com.amazonaws.services.s3.AmazonS3;
import io.micrometer.common.util.StringUtils;
import jakarta.annotation.Nonnull;
import java.util.Date;
import java.util.Objects;
import java.util.Random;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.dto.ImageDto;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@RequiredArgsConstructor
@Slf4j
@Service
public class S3FileService {

    private final AmazonS3 amazonS3;

    @Value("${cloud.aws.s3.bucket}")
    private String bucket;

    public ImageDto getPreSignedUrlForUpload(@Nonnull String extension) {
        String filename = createPath(extension);

        Date expiration = getPreSignedUrlExpiration();

        String imageURL = amazonS3.generatePresignedUrl(bucket, filename, expiration,
                HttpMethod.PUT)
            .toString();

        return ImageDto.from(imageURL, filename);
    }

    public String getMemberPreSignedUrlForLoad(String gender, String filename) {
        Date expiration = getPreSignedUrlExpiration();

        log.info("getMemberPreSignedUrlForLoad: {}", filename);

        if (filename.contains("default.png") && Objects.equals(gender, "MALE")) {
            log.info("getMemberPreSignedUrlForLoad: {}", filename);
            return amazonS3.generatePresignedUrl(bucket, "images/maleDefault.webp", expiration)
                .toString();
        }

        if (filename.contains("default.png") && Objects.equals(gender, "FEMALE")) {
            return amazonS3.generatePresignedUrl(bucket, "images/femaleDefault.webp", expiration)
                .toString();
        }

        return amazonS3.generatePresignedUrl(bucket, filename, expiration)
            .toString();
    }


    public String getPreSignedUrlForLoad(String filename) {

        log.info("getPreSignedUrlForLoad: {}", filename);
        
        Date expiration = getPreSignedUrlExpiration();

        if (filename.contains("default.png")) {
            return amazonS3.generatePresignedUrl(bucket, "images/default.png", expiration)
                .toString();
        }

        // TODO: 더미 데이터를 위한 코드 실제 서비스에서는 제거할 것
        if (filename.contains("test")) {
            Random random = new Random();
            switch (random.nextInt(4) + 1) {
                case 1:
                    return amazonS3.generatePresignedUrl(bucket, "images/defaultroom1.jpg",
                            expiration)
                        .toString();
                case 2:
                    return amazonS3.generatePresignedUrl(bucket, "images/defaultroom2.jpg",
                            expiration)
                        .toString();
                case 3:
                    return amazonS3.generatePresignedUrl(bucket, "images/defaultroom3.jpg",
                            expiration)
                        .toString();
                case 4:
                    return amazonS3.generatePresignedUrl(bucket, "images/defaultroom.jpeg",
                            expiration)
                        .toString();
            }
        }

        return amazonS3.generatePresignedUrl(bucket, filename, expiration)
            .toString();
    }

    private Date getPreSignedUrlExpiration() {
        Date expiration = new Date();
        long expTimeMillis = expiration.getTime();
        expTimeMillis += 1000 * 60 * 2;
        expiration.setTime(expTimeMillis);
        return expiration;
    }

    /*
    이 id 로 image table에서 id 로 활용
     */
    private String createFileId() {
        return UUID.randomUUID().toString();
    }

    private String createPath(String extension) {
        String fileId = createFileId();
        return String.format("%s/%s", "images", fileId + extension);
    }
}
