package org.capstone.maru.service;

import java.util.List;
import java.util.Optional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.MemberCard;
import org.capstone.maru.domain.ProfileImage;
import org.capstone.maru.dto.MemberCardDto;
import org.capstone.maru.dto.MemberProfileDto;
import org.capstone.maru.dto.response.AuthResponse;
import org.capstone.maru.repository.MemberCardRepository;
import org.capstone.maru.repository.ProfileImageRepository;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;


@RequiredArgsConstructor
@Service
@Slf4j
public class ProfileService {

    private final MemberAccountService memberAccountService;

    private final S3FileService s3FileService;

    private final MemberCardRepository memberCardRepository;

    private final ProfileImageRepository profileImageRepository;

    @Transactional
    public MemberCardDto updateMyCard(String memberId, Long cardId, String location,
        List<String> features) {
        log.info("updateMyCard - memberId: {}, myFeatures: {}", memberId, features);

        MemberAccount memberAccount = memberAccountService.searchMemberAccount(memberId);
        MemberCard memberCard = memberCardRepository.findById(cardId)
            .orElseThrow(() -> new IllegalArgumentException("invaild cardId"));

        MemberCard myCard = memberAccount.getMyCard();
        MemberCard mateCard = memberAccount.getMateCard();

        if (!myCard.equals(memberCard) && !mateCard.equals(memberCard)) {
            throw new IllegalArgumentException("MemberCard not found");
        }

        memberCard.updateLocation(location);
        memberCard.updateMemberFeatures(features);

        return MemberCardDto.from(memberCard);
    }

    @Transactional(readOnly = true)
    public MemberProfileDto getMemberProfile(String memberId) {

        log.info("getMyCard - memberId: {}", memberId);

        MemberAccount memberAccount = memberAccountService.searchMemberAccount(memberId);
        MemberCard myCard = memberAccount.getMyCard();
        MemberCard mateCard = memberAccount.getMateCard();
        ProfileImage profileImage = memberAccount.getProfileImage();

        log.info("myCard: {}", myCard.getMemberFeatures());
        log.info("mateCard: {}", mateCard.getMemberFeatures());
        log.info("profileImage: {}", profileImage.getFileName());

        String imgURL = s3FileService.getPreSignedUrlForLoad(profileImage.getFileName());

        AuthResponse authResponse = AuthResponse.from(memberAccount);

        return MemberProfileDto.from(imgURL, myCard, mateCard, authResponse);
    }

    @Transactional
    public MemberCardDto updateRoomCard(String memberId, String roomCardId, List<String> strings) {
        log.info("updateRoomCard - memberId: {}, roomCardId: {}, myFeatures: {}", memberId,
            roomCardId, strings);

        /*
         * TODO
         * 해당 roomCardId의 글이 로그인 한 사용자가 쓴 일치하는지 확인
         * 해당 roomCardId의 글이 존재하는지 확인, 존재하는 경우 그 카드를 가져옵니다.
         * 가져온 카드의 정보를 업데이트하고 저장합니다.
         * */

        return MemberCardDto.builder().build();
    }

    @Transactional
    public MemberCardDto getCard(Long cardId) {
        log.info("getCard - cardId: {}", cardId);

        MemberCard memberCard = memberCardRepository.findById(cardId)
            .orElseThrow(() -> new IllegalArgumentException("MemberCard not found"));

        return MemberCardDto.from(memberCard);

    }

    /*
     * 프로필 이미지 업데이트
     * 이때는 이미지가 이미 업로드 되어 있는 시점이다.
     */
    @Transactional
    public void updateProfileImage(String memberId, String fileName) {
        log.info("updateProfileImage - memberId: {}, fileName: {}", memberId, fileName);

        MemberAccount memberAccount = memberAccountService.searchMemberAccount(memberId);

        Optional<ProfileImage> profileImage = profileImageRepository.findById(fileName);

        if (profileImage.isEmpty()) {
            throw new IllegalArgumentException("ProfileImage not found");
        }

        memberAccount.updateProfileImage(profileImage.get());
    }
}
