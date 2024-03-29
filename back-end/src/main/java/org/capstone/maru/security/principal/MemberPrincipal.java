package org.capstone.maru.security.principal;

import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.Getter;
import org.capstone.maru.dto.MemberAccountDto;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.oauth2.core.user.OAuth2User;

public record MemberPrincipal(
    String memberId,
    String email,
    String nickname,
    String birthYear,
    String gender,
    String phoneNumber,
    Collection<? extends GrantedAuthority> authorities,
    Map<String, Object> oAuth2Attributes
) implements UserDetails, OAuth2User {

    public static MemberPrincipal of(
        String memberId,
        String email,
        String nickname,
        String birthYear,
        String gender,
        String phoneNumber,
        Map<String, Object> oAuth2Attributes
    ) {
        Set<RoleType> roleTypes = Set.of(RoleType.MEMBER);
        return of(
            memberId,
            email,
            nickname,
            birthYear,
            gender,
            phoneNumber,
            roleTypes
                .stream()
                .map(RoleType::getName)
                .map(SimpleGrantedAuthority::new)
                .collect(Collectors.toUnmodifiableSet()),
            oAuth2Attributes
        );
    }

    public static MemberPrincipal of(
        String memberId,
        String email,
        String nickname,
        String birthYear,
        String gender,
        String phoneNumber,
        Collection<? extends GrantedAuthority> authorities,
        Map<String, Object> oAuth2Attributes
    ) {
        return new MemberPrincipal(
            memberId,
            email,
            nickname,
            birthYear,
            gender,
            phoneNumber,
            authorities,
            oAuth2Attributes
        );
    }

    /**
     * MemberAccountDto를 받아서 현재 객체를 리턴합니다.
     *
     * @param dto
     * @return
     */
    public static MemberPrincipal from(MemberAccountDto dto,
        Map<String, Object> oAuth2Attributes) {
        return MemberPrincipal.of(
            dto.memberId(),
            dto.email(),
            dto.nickname(),
            dto.birthYear(),
            dto.gender(),
            dto.phoneNumber(),
            oAuth2Attributes
        );
    }

    /**
     * 현재 객체를 dto로 바꿔줍니다
     *
     * @return
     */
    public MemberAccountDto toDto() {
        return MemberAccountDto.builder()
                               .memberId(memberId)
                               .email(email)
                               .nickname(nickname)
                               .build();
    }

    // -- OAuth2User -- //
    @Override
    public String getName() {
        return memberId;
    }

    @Override
    public Map<String, Object> getAttributes() {
        return oAuth2Attributes;
    }

    // -- UserDetails -- //
    @Override
    public String getUsername() {
        return memberId;
    }

    @Override
    public String getPassword() {
        return null;
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return authorities;
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        return true;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }

    public enum RoleType {
        MEMBER("ROLE_MEMBER");

        @Getter
        private final String name;

        RoleType(String name) {
            this.name = name;
        }
    }
}
