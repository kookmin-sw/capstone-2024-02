package org.capstone.maru.controller;


import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.security.token.TokenProvider;
import org.capstone.maru.service.FirestoreService;
import org.capstone.maru.service.S3FileService;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@Slf4j
@RequiredArgsConstructor
@RestController
public class MainController {

    private final TokenProvider tokenProvider;

    private final FirestoreService firestoreService;

    @GetMapping("/")
    public String root() {
        return "health check is running!";
    }

    @GetMapping("/test")
    public String test(@AuthenticationPrincipal MemberPrincipal memberPrincipal) {

        return String.join(
            " ",
            List.of(
                memberPrincipal.getName(),
                memberPrincipal.email(),
                memberPrincipal.birthYear(),
                memberPrincipal.gender(),
                memberPrincipal.phoneNumber()
            )
        );
    }

    @GetMapping("/test2")
    public void test2() {
        firestoreService.printAllDocumentsInCollection();
    }

}
