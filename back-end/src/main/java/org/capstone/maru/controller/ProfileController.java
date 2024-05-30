package org.capstone.maru.controller;

import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.dto.FollowingDto;
import org.capstone.maru.dto.FeatureCardDto;
import org.capstone.maru.dto.MemberProfileDto;
import org.capstone.maru.dto.request.CertifyUnivRequest;
import org.capstone.maru.dto.request.EmailSearchRequest;
import org.capstone.maru.dto.SimpleMemberCardDto;
import org.capstone.maru.dto.request.MemberFeatureRequest;
import org.capstone.maru.dto.request.MemberIdRequest;
import org.capstone.maru.dto.request.SettingRequest;
import org.capstone.maru.dto.response.APIResponse;
import org.capstone.maru.dto.response.SimpleMemberCardResponse;
import org.capstone.maru.dto.response.SimpleMemberProfileResponse;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.service.CertificateService;
import org.capstone.maru.service.FollowService;
import org.capstone.maru.service.ProfileService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RestController
@RequestMapping("/profile")
public class ProfileController {

    private final ProfileService profileService;
    private final FollowService followService;
    private final CertificateService certificateService;

    /*
     * 이메일로 사용자 프로필 검색
     */
    @PostMapping("/search")
    public ResponseEntity<APIResponse> searchProfile(
        @RequestBody EmailSearchRequest email
    ) {
        log.info("call searchProfile : {}", email.email());

        SimpleMemberProfileResponse result = profileService.searchProfile(email.email());
        return ResponseEntity.ok(APIResponse.success(result));
    }

    /*
     * 내 프로필 수정
     */
    @PutMapping("/{cardId}")
    public ResponseEntity<APIResponse> updateMemberCard(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @PathVariable Long cardId,
        @RequestBody MemberFeatureRequest memberFeatureRequest
    ) {
        log.info("call updateProfile : {}", memberFeatureRequest);

        String memberId = memberPrincipal.memberId();

        FeatureCardDto result = profileService.updateMyCard(
            memberId,
            cardId,
            memberFeatureRequest.location(),
            memberFeatureRequest.features()
        );

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @PostMapping
    public ResponseEntity<APIResponse> getMemberProfile(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @RequestBody MemberIdRequest request
    ) {
        log.info("call getProfile : {}", request.memberId());

        MemberProfileDto result = profileService.getMemberProfile(request.memberId(),
            memberPrincipal.gender());

        return ResponseEntity.ok(APIResponse.success(result));
    }


    @GetMapping("/card/{cardId}")
    public ResponseEntity<APIResponse> getCardData(
        @PathVariable Long cardId
    ) {
        log.info("call getCardData : {}", cardId);
        FeatureCardDto result = profileService.getCard(cardId);

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @PutMapping("/room/{roomCardId}")
    public ResponseEntity<APIResponse> updateRoomCardProfile(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @PathVariable String roomCardId,
        @RequestBody MemberFeatureRequest memberFeatureRequest
    ) {
        log.info("call updateRoomCardProfile : {}", memberFeatureRequest);
        String memberId = memberPrincipal.memberId();

        FeatureCardDto result = profileService.updateRoomCard(memberId, roomCardId,
            memberFeatureRequest.features());

        return ResponseEntity.ok(APIResponse.success(result));
    }

    /*
     * 팔로우
     */
    @PostMapping("/follow")
    public ResponseEntity<APIResponse> follow(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @RequestBody MemberIdRequest request
    ) {
        followService.followUser(memberPrincipal.memberId(), request.memberId());
        return ResponseEntity.ok(APIResponse.success());
    }

    /*
     * 팔로우 취소
     */
    @PostMapping("/unfollow")
    public ResponseEntity<APIResponse> unfollow(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @RequestBody MemberIdRequest request
    ) {
        followService.unfollowUser(memberPrincipal.memberId(), request.memberId());
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

    @GetMapping("/mutual/follow")
    public ResponseEntity<APIResponse> mutualFollowings(
        @AuthenticationPrincipal MemberPrincipal principal) {
        FollowingDto result = followService.getMutualFollower(principal.memberId());
        return ResponseEntity.ok(APIResponse.success(result));
    }

    /*
     * 프로필 이미지 수정
     */
    @PatchMapping("/image")
    public ResponseEntity<APIResponse> updateProfileImage(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @RequestBody String fileName
    ) {
        log.info("call updateProfileImage : {}", fileName);

        profileService.updateProfileImage(memberPrincipal.memberId(), fileName);

        return ResponseEntity.ok(APIResponse.success());
    }

    /*
     * 단어가 포함된 사용자 검색 by email
     */
    @PostMapping("/search/contain")
    public ResponseEntity<APIResponse> searchProfileByEmail(
        @RequestBody String email
    ) {
        log.info("call searchProfileByEmail : {}", email);

        List<SimpleMemberProfileResponse> result = profileService.searchContainByEmail(email);
        return ResponseEntity.ok(APIResponse.success(result));
    }

    @PostMapping("/certificate")
    public ResponseEntity<APIResponse> certifySchoolAndEmail(
        @AuthenticationPrincipal MemberPrincipal principal,
        @RequestBody CertifyUnivRequest certifyUnivRequest
    ) {
        if (certifyUnivRequest.code() == null) {
            certificateService.certifyUnivAndEmail(
                certifyUnivRequest.email(),
                certifyUnivRequest.univName()
            );

            return ResponseEntity.ok(APIResponse.success());
        }

        certificateService.certifyCode(
            principal.memberId(),
            certifyUnivRequest.email(),
            certifyUnivRequest.univName(),
            certifyUnivRequest.code()
        );

        return ResponseEntity.ok(APIResponse.success());
    }

    /*
    내 메이트 카드 기반 유저 추천 리스트 조회
     */
    @GetMapping("/recommend")
    public ResponseEntity<APIResponse> getRecommendMateCard(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @RequestParam(name = "cardOption") String cardOption
    ) {
        log.info("call getRecommendMateCard : {}", memberPrincipal.memberId());

        List<SimpleMemberCardDto> recommendMember = profileService.getRecommendMember(
            memberPrincipal.memberId(), memberPrincipal.gender(), cardOption);

        List<SimpleMemberCardResponse> result = recommendMember.stream()
            .map(SimpleMemberCardResponse::from)
            .toList();

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @PatchMapping("/setting")
    public ResponseEntity<APIResponse> updateProfileSetting(
        @AuthenticationPrincipal MemberPrincipal memberPrincipal,
        @RequestBody SettingRequest settingRequest
    ) {
        profileService.updateRecommendOnOff(memberPrincipal.memberId(),
            settingRequest.recommendOn());
        return ResponseEntity.ok(APIResponse.success());
    }

}
