package org.capstone.maru.service;

import com.amazonaws.HttpMethod;
import com.amazonaws.services.s3.AmazonS3;
import io.micrometer.common.util.StringUtils;
import jakarta.annotation.Nonnull;
import java.util.Date;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.capstone.maru.dto.ImageDto;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@RequiredArgsConstructor
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

    public String getPreSignedUrlForLoad(String filename) {

        Date expiration = getPreSignedUrlExpiration();

        if (filename.contains("default.png")) {
            return amazonS3.generatePresignedUrl(bucket, "images/default.png", expiration)
                .toString();
        }
        if (StringUtils.isNotBlank(filename)) {
            filename = "images/" + filename;
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
