package org.capstone.maru.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.dto.ImageDto;
import org.capstone.maru.dto.response.APIResponse;
import org.capstone.maru.service.S3FileService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RestController
@RequestMapping("/image")
public class ImageController {

    private final S3FileService s3FileService;

    @GetMapping("/upload")
    public ResponseEntity<APIResponse> uploadFile(@RequestParam("extension") String extension) {
        ImageDto data = s3FileService.getPreSignedUrlForUpload(extension);

        return ResponseEntity.ok(APIResponse.success(data));
    }

    @GetMapping("/load")
    public ResponseEntity<APIResponse> loadFile(@RequestParam("fileName") String fileName) {
        String preSignedUrl = s3FileService.getPreSignedUrlForLoad(fileName);

        return ResponseEntity.ok(APIResponse.success(preSignedUrl));
    }

}
