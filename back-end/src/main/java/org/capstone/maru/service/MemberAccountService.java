package org.capstone.maru.service;

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.Follow;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.domain.MemberRoom;
import org.capstone.maru.domain.ProfileImage;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.security.exception.MemberAccountExistentException;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.dto.MemberAccountDto;
import org.capstone.maru.repository.postgre.MemberAccountRepository;
import org.capstone.maru.security.exception.MemberAccountNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Mono;

@Slf4j
@RequiredArgsConstructor
@Transactional
@Service
public class MemberAccountService {

    private final MemberAccountRepository memberAccountRepository;

//    private final RecommendService recommendService;

    @Transactional(readOnly = true)
    public MemberAccountDto searchMember(String memberId) {
        Optional<MemberAccountDto> memberAccount = memberAccountRepository
            .findById(memberId)
            .map(MemberAccountDto::from);

        if (memberAccount.isEmpty()) {
            throw new MemberAccountNotFoundException(RestErrorCode.MEMBER_NOT_FOUND);
        }

        return memberAccount.get();
    }

    @Transactional(readOnly = true)
    public MemberAccount searchMemberAccount(String memberId) {
        Optional<MemberAccount> memberAccount = memberAccountRepository.findById(memberId);

        if (memberAccount.isEmpty()) {
            log.info("MemberAccount not found: {}", memberId);
            throw new MemberAccountNotFoundException(RestErrorCode.MEMBER_NOT_FOUND);
        }

        return memberAccount.get();
    }

    @Transactional
    public MemberAccountDto login(
        String memberId,
        String email,
        String nickname,
        String birthYear,
        String gender,
        String phoneNumber
    ) {
        Optional<MemberAccount> memberAccount = memberAccountRepository.findByEmail(email);

        if (memberAccount.isEmpty()) {

            FeatureCard myCard = FeatureCard.of(null, null);
            FeatureCard mateCard = FeatureCard.of(null, null);

            Set<Follow> followers = new HashSet<>();
            Set<Follow> followings = new HashSet<>();

            ProfileImage profileImage = ProfileImage.defaultImage(memberId);

            List<MemberRoom> chatRooms = List.of();

            MemberAccount member = MemberAccount.of(
                memberId,
                email,
                nickname,
                birthYear,
                gender,
                phoneNumber,
                memberId,
                true,
                myCard,
                mateCard,
                followers,
                followings,
                profileImage,
                chatRooms
            );

            MemberAccount newMemberAccount = memberAccountRepository.save(member);

            return MemberAccountDto.from(newMemberAccount);
        }

        if (memberAccount.get().getMemberId().equals(memberId)) {
            return MemberAccountDto.from(memberAccount.get());
        }

        throw new MemberAccountExistentException(RestErrorCode.DUPLICATE_VALUE);
    }

    /*
    최초 로그인인지 판단
     */
    public Boolean isInitialized(String memberId) {
        return memberAccountRepository.getInitializedById(memberId).getInitialized();
    }

    /*
     * 이메일로 회원 찾기
     */
    public MemberAccount searchMemberAccountByEmail(String email) {
        return memberAccountRepository.findByEmail(email)
                                      .orElseThrow(() -> new MemberAccountNotFoundException(
                                          RestErrorCode.MEMBER_NOT_FOUND));
    }

    public List<MemberAccount> searchContainByEmail(String word) {
        return memberAccountRepository.findByEmailContaining(word);
    }
}
