package org.capstone.maru.controller;


import java.util.List;
import lombok.RequiredArgsConstructor;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.security.token.TokenProvider;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RequiredArgsConstructor
@RestController
public class MainController {

    private final TokenProvider tokenProvider;

    @GetMapping("/")
    public String root() {
        return "health check!";
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
}
