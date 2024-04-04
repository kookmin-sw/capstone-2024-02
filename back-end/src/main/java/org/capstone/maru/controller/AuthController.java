package org.capstone.maru.controller;

import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.dto.response.APIResponse;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.dto.response.AuthResponse;
import org.capstone.maru.security.token.TokenDto;
import org.capstone.maru.security.token.TokenReIssuer;
import org.capstone.maru.service.MemberAccountService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RestController
@RequestMapping("/auth")
public class AuthController {

    private final TokenReIssuer tokenReIssuer;
    private final MemberAccountService memberAccountService;

    @PostMapping("/token/refresh")
    public ResponseEntity<APIResponse> refreshToken(HttpServletRequest request) {
        TokenDto newAccessToken = tokenReIssuer.reissueTokens(request);
        return ResponseEntity.ok(APIResponse.success(newAccessToken));
    }

    /**
     * 로그인한 사용자의 정보를 반환합니다. 이는 첫 로그인시에 요청하므로, 사용자의 정보와 첫 로그인 인지를 반환합니다.
     *
     * @param memberPrincipal 로그인한 사용자의 정보
     * @return 로그인한 사용자의 정보
     */
    @GetMapping("/initial/info")
    public ResponseEntity<APIResponse> initialInfo(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal) {
        MemberAccount memberAccount = memberAccountService.searchMemberAccount(
            memberPrincipal.memberId());
        AuthResponse data = AuthResponse.from(memberAccount);
        return ResponseEntity.ok(APIResponse.success(data));
    }

    /**
     * 로그인한 사용자의 정보를 반환합니다. 이는 첫 로그인시에 요청하므로, 사용자의 정보와 첫 로그인 인지를 반환합니다. 로그인 사용자 정보만을 요청시는 이미 프로필을
     * 작성했다고 가정하고 initialized를 false로 반환합니다.
     *
     * @param memberPrincipal 로그인한 사용자의 정보
     * @return 로그인한 사용자의 정보
     */
    @GetMapping("/info")
    public ResponseEntity<APIResponse> info(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal) {
        AuthResponse data = AuthResponse.from(memberPrincipal, false);
        return ResponseEntity.ok(APIResponse.success(data));
    }

}
