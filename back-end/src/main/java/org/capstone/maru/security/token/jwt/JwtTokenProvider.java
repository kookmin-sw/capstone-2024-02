package org.capstone.maru.security.token.jwt;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.ExpiredJwtException;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.MalformedJwtException;
import io.jsonwebtoken.UnsupportedJwtException;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import io.jsonwebtoken.security.SecurityException;
import io.jsonwebtoken.security.SignatureException;
import java.security.Key;
import java.util.Date;
import java.util.List;
import java.util.Map;
import javax.crypto.SecretKey;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.security.exception.InvalidTokenException;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.security.response.OAuth2Response;
import org.capstone.maru.security.response.OAuth2ResponseFactory;
import org.capstone.maru.security.token.RefreshTokenService;
import org.capstone.maru.security.token.TokenDto;
import org.capstone.maru.security.token.TokenProvider;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.oauth2.client.authentication.OAuth2AuthenticationToken;
import org.springframework.stereotype.Component;

@Slf4j
@Component("jwtTokenProvider")
public class JwtTokenProvider implements TokenProvider, InitializingBean {

    private final RefreshTokenService refreshTokenService;

    private final String SECRET_KEY_STRING;
    private Key SECRET_KEY;
    private final Long ACCESS_TOKEN_VALID_MILLI_SECOND = 60 * 60 * 1000L;
    private final Long REFRESH_TOKEN_VALID_MILLI_SECOND = 7 * 24 * 60 * 60 * 1000L; // 7 days
    private static final String AUTHORITY = "authorities";
    private static final String EMAIL = "email";
    private static final String NICKNAME = "nickname";
    private static final String BIRTH_YEAR = "birthyear";
    private static final String GENDER = "gender";
    private static final String PHONE_NUMBER = "phonenumber";

    public JwtTokenProvider(
        @Value("${jwt.secret}") String SECRET_KEY_STRING,
        @Autowired RefreshTokenService refreshTokenService
    ) {
        this.SECRET_KEY_STRING = SECRET_KEY_STRING;
        this.refreshTokenService = refreshTokenService;
    }

    @Override
    public void afterPropertiesSet() throws Exception {
        byte[] keyBytes = Decoders.BASE64.decode(SECRET_KEY_STRING);
        this.SECRET_KEY = Keys.hmacShaKeyFor(keyBytes);
    }

    @Override
    public boolean validate(String token) {
        try {
            Jwts.parser().verifyWith((SecretKey) SECRET_KEY).build().parseSignedClaims(token);
            return true;
        } catch (SignatureException ex) {
            log.error("JWT signature does not match");
        } catch (SecurityException ex) {
            log.error("Invalid JWT signature");
        } catch (MalformedJwtException ex) {
            log.error("Invalid JWT token");
        } catch (ExpiredJwtException ex) {
            log.error("Expired JWT token");
        } catch (UnsupportedJwtException ex) {
            log.error("Unsupported JWT token");
        } catch (IllegalArgumentException ex) {
            log.error("JWT claims string is empty.");
        }
        return false;
    }

    @Override
    @SuppressWarnings("unchecked")
    public OAuth2AuthenticationToken decode(String token) {
        Claims claims = this.parseClaims(token);
        String memberId = claims.getSubject();
        String email = claims.get(EMAIL, String.class);
        String nickname = claims.get(NICKNAME, String.class);
        String birthYear = claims.get(BIRTH_YEAR, String.class);
        String gender = claims.get(GENDER, String.class);
        String phoneNumber = claims.get(PHONE_NUMBER, String.class);
        List<? extends GrantedAuthority> grantedAuthorities = claims
            .get(AUTHORITY, List.class)
            .stream()
            .map(auth -> new SimpleGrantedAuthority((String) auth))
            .toList();

        log.info("[Debug] decode token memberId: {}", memberId);
        MemberPrincipal principal = MemberPrincipal.of(
            memberId,
            email,
            nickname,
            birthYear,
            gender,
            phoneNumber,
            grantedAuthorities,
            Map.of()
        );

        return new OAuth2AuthenticationToken(principal, grantedAuthorities, "maru");
    }

    @Override
    public String createAccessToken(Authentication authentication) {
        return createToken(authentication, ACCESS_TOKEN_VALID_MILLI_SECOND);
    }

    @Override
    public String createRefreshToken(Authentication authentication) {
        String token = this.createToken(authentication, REFRESH_TOKEN_VALID_MILLI_SECOND);
        refreshTokenService.saveRefreshToken(token, REFRESH_TOKEN_VALID_MILLI_SECOND);
        return token;
    }

    @Override
    public TokenDto reissueAccessTokenUsing(String refreshToken) throws InvalidTokenException {
        try {
            Claims claims = this.parseClaims(refreshToken);
            return this.createAccessTokenOnly(claims);
        } catch (JwtException e) {
            throw new InvalidTokenException(RestErrorCode.INVALID_TOKEN_VALUE, "리프레쉬 토큰이 만료되었습니다.");
        }
    }

    @Override
    public TokenDto reissueTokensUsing(String refreshToken) throws JwtException {
        try {
            Claims claims = this.parseClaims(refreshToken);
            return this.createTokens(claims);
        } catch (JwtException e) {
            throw new InvalidTokenException(RestErrorCode.INVALID_TOKEN_VALUE, "리프레쉬 토큰이 만료되었습니다.");
        }
    }

    @Override
    public Long getExpiration() {
        return this.ACCESS_TOKEN_VALID_MILLI_SECOND / 1000;
    }

    // -- JWT 생성, 리프레쉬, 파싱 로직 -- //
    private String createToken(Authentication authentication, Long expiration) {
        log.info("[Debug] creating token!");

        assert authentication != null;

        OAuth2Response oAuth2Response = OAuth2ResponseFactory.getOAuth2Response(
            (OAuth2AuthenticationToken) authentication
        );

        String memberId =
            ((OAuth2AuthenticationToken) authentication).getAuthorizedClientRegistrationId()
                + "_" + oAuth2Response.id();

        List<String> authorities = authentication.getAuthorities()
                                                 .stream()
                                                 .map(GrantedAuthority::getAuthority)
                                                 .toList();

        Date now = new Date();
        Date expiryDate = new Date(now.getTime() + expiration);

        return Jwts.builder()
                   .subject(memberId)
                   .claim(EMAIL, oAuth2Response.email())
                   .claim(NICKNAME, oAuth2Response.nickname())
                   .claim(BIRTH_YEAR, oAuth2Response.birthYear())
                   .claim(GENDER, oAuth2Response.gender().name())
                   .claim(PHONE_NUMBER, oAuth2Response.phoneNumber())
                   .claim(AUTHORITY, authorities)
                   .issuedAt(now)
                   .expiration(expiryDate)
                   .signWith(SECRET_KEY)
                   .compact();
    }

    private TokenDto createAccessTokenOnly(Claims claims) {
        Date now = new Date();

        String accessToken = Jwts.builder()
                                 .claims(claims)
                                 .issuedAt(now)
                                 .expiration(
                                     new Date(now.getTime() + ACCESS_TOKEN_VALID_MILLI_SECOND))
                                 .signWith(SECRET_KEY)
                                 .compact();

        return TokenDto.builder()
                       .grantType("Bearer")
                       .accessToken(accessToken)
                       .refreshToken("")
                       .accessTokenExpireDate(getExpiration())
                       .build();
    }

    private TokenDto createTokens(Claims claims) {
        Date now = new Date();

        String accessToken = Jwts.builder()
                                 .claims(claims)
                                 .issuedAt(now)
                                 .expiration(
                                     new Date(now.getTime() + ACCESS_TOKEN_VALID_MILLI_SECOND))
                                 .signWith(SECRET_KEY)
                                 .compact();

        String refreshToken = Jwts.builder()
                                  .claims(claims)
                                  .issuedAt(now)
                                  .expiration(
                                      new Date(now.getTime() + REFRESH_TOKEN_VALID_MILLI_SECOND))
                                  .signWith(SECRET_KEY)
                                  .compact();

        return TokenDto.builder()
                       .grantType("Bearer")
                       .accessToken(accessToken)
                       .refreshToken(refreshToken)
                       .accessTokenExpireDate(getExpiration())
                       .build();
    }

    private Claims parseClaims(String token) throws JwtException {
        return Jwts.parser()
                   .verifyWith((SecretKey) SECRET_KEY)
                   .build()
                   .parseSignedClaims(token)
                   .getPayload();
    }
}
