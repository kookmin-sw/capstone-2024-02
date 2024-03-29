package org.capstone.maru.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.dto.FollowingDto;
import org.capstone.maru.dto.MemberCardDto;
import org.capstone.maru.dto.MemberProfileDto;
import org.capstone.maru.dto.request.MemberFeatureRequest;
import org.capstone.maru.dto.response.APIResponse;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.service.FollowService;
import org.capstone.maru.service.ProfileService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RestController
@RequestMapping("/profile")
public class ProfileController {

    private final ProfileService profileService;
    private final FollowService followService;

    @PutMapping
    public ResponseEntity<APIResponse> updateMyProfile(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @RequestBody MemberFeatureRequest memberFeatureRequest
    ) {
        log.info("call updateProfile : {}", memberFeatureRequest);

        String memberId = memberPrincipal.memberId();
        MemberCardDto result = profileService.updateMyCard(memberId,
            memberFeatureRequest.myFeatures());

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @GetMapping("/{memberId}")
    public ResponseEntity<APIResponse> getMemberProfile(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @PathVariable String memberId
    ) {
        log.info("call getProfile : {}", memberId);
        MemberProfileDto result = profileService.getMemberCard(memberId, memberPrincipal);

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @PutMapping("/{roomCardId}")
    public ResponseEntity<APIResponse> updateRoomCardProfile(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @PathVariable String roomCardId,
        @RequestBody MemberFeatureRequest memberFeatureRequest
    ) {
        log.info("call updateRoomCardProfile : {}", memberFeatureRequest);
        String memberId = memberPrincipal.memberId();

        MemberCardDto result = profileService.updateRoomCard(memberId, roomCardId,
            memberFeatureRequest.myFeatures());

        return ResponseEntity.ok(APIResponse.success(result));
    }

    /*
     * 팔로우
     */
    @PostMapping("/{memberId}/follow")
    public ResponseEntity<APIResponse> follow(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @PathVariable String memberId
    ) {
        followService.followUser(memberPrincipal.memberId(), memberId);
        return ResponseEntity.ok(APIResponse.success());
    }

    /*
     * 내가 팔로잉한 사람 조회
     */
    @GetMapping("/follow")
    public ResponseEntity<APIResponse> getFollowing(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal
    ) {
        FollowingDto result = followService.getFollowings(memberPrincipal.memberId());

        return ResponseEntity.ok(APIResponse.success(result));
    }
}