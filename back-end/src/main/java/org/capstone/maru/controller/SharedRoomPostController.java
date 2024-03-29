package org.capstone.maru.controller;

import static org.capstone.maru.security.cookie.HttpCookieOAuth2AuthorizationRequestRepository.REDIRECT_URL_PARAM_COOKIE_NAME;

import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Set;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.annotation.RequestQueryString;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.RoomInfoDto;
import org.capstone.maru.dto.StudioRoomPostDto;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.capstone.maru.dto.request.StudioRoomPostRequest;
import org.capstone.maru.dto.response.APIResponse;
import org.capstone.maru.dto.response.StudioRoomPostDetailResponse;
import org.capstone.maru.dto.response.StudioRoomPostResponse;
import org.capstone.maru.security.cookie.CookieUtils;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.service.SharedRoomPostService;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort.Direction;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RestController
@RequestMapping("/shared/posts")
public class SharedRoomPostController {

    private final SharedRoomPostService sharedRoomPostService;

    @GetMapping("/studio")
    public ResponseEntity<APIResponse> studioRoomPosts(
        @AuthenticationPrincipal MemberPrincipal principal,
        @RequestQueryString(name = "filter", required = false) SearchFilterRequest searchFilterRequest,
        @RequestParam(name = "search", required = false) String searchKeyWords,
        @PageableDefault(size = 10, sort = "createdAt", direction = Direction.DESC) Pageable pageable
    ) {
        Page<StudioRoomPostResponse> result = sharedRoomPostService
            .searchStudioRoomPosts(
                principal.gender(),
                searchFilterRequest,
                searchKeyWords,
                pageable
            )
            .map(StudioRoomPostResponse::from);

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @GetMapping("/studio/{postId}")
    public ResponseEntity<APIResponse> studioRoomPostDetail(
        @AuthenticationPrincipal MemberPrincipal principal,
        @PathVariable("postId") Long postId
    ) {
        StudioRoomPostDetailResponse result = StudioRoomPostDetailResponse.from(
            sharedRoomPostService.getStudioRoomPostDetail(postId, principal.gender())
        );

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @PostMapping(path = "/studio", consumes = {MediaType.MULTIPART_FORM_DATA_VALUE})
    public void postNewStudioRoomPost(
        @AuthenticationPrincipal MemberPrincipal principal,
        @ModelAttribute StudioRoomPostRequest studioRoomPostRequest,
        HttpServletRequest request, HttpServletResponse response
    ) throws IOException {
        StudioRoomPostDto studioRoomPostDto = studioRoomPostRequest.toBaseStudioRoomPostDto();
        RoomInfoDto roomInfoDto = studioRoomPostRequest.toRoomInfoDto();
        Set<RoomImageDto> roomImagesDto = studioRoomPostRequest.toRoomImagesDto();

        sharedRoomPostService.saveStudioRoomPost(
            principal.memberId(), studioRoomPostDto, roomImagesDto, roomInfoDto
        );

        // 성공하면 프론트에서 redirect 하라는 곳으로 redirect 시키기
        String redirectURL = CookieUtils
            .resolveCookie(request, REDIRECT_URL_PARAM_COOKIE_NAME)
            .map(Cookie::getValue)
            .orElse("/");

        response.sendRedirect(redirectURL);
    }

    @DeleteMapping("/studio/{postId}")
    public void deleteStudioRoomPost(
        @AuthenticationPrincipal MemberPrincipal principal,
        @PathVariable("postId") Long postId,
        HttpServletRequest request, HttpServletResponse response
    ) throws IOException {
        sharedRoomPostService.deleteStudioRoomPost(postId, principal.memberId());

        String redirectURL = CookieUtils
            .resolveCookie(request, REDIRECT_URL_PARAM_COOKIE_NAME)
            .map(Cookie::getValue)
            .orElse("/");

        response.sendRedirect(redirectURL);
    }
}
