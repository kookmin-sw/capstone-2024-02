package org.capstone.maru.controller;


import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.security.token.TokenProvider;
import org.capstone.maru.service.RecommendService;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RestController
public class MainController {

    private final TokenProvider tokenProvider;

    private final RecommendService recommendService;

    @GetMapping("/")
    public String root() {
        return "health check";
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
        recommendService.updateRecommendation(
            "naver_htT4VdDRPKqGqKpnncpa71HCA4CVg5LdRC1cWZhCnF8", "my",
            "post"
        ).subscribe();
    }

}
