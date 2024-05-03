package org.capstone.maru.security.config;

import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.security.cookie.HttpCookieOAuth2AuthorizationRequestRepository;
import org.capstone.maru.security.filter.TokenAuthenticationProcessingFilter;
import org.capstone.maru.security.service.CustomOAuth2UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.security.servlet.PathRequest;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.AuthenticationEntryPoint;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.AuthenticationFailureHandler;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.security.web.authentication.logout.LogoutHandler;
import org.springframework.security.web.authentication.logout.LogoutSuccessHandler;

@Slf4j
@Order(0)
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    private final AuthenticationEntryPoint authEntryPoint;
    private final AuthenticationFailureHandler authFailureHandler;
    private final AuthenticationSuccessHandler authSuccessHandler;
    private final TokenAuthenticationProcessingFilter tokenAuthenticationProcessingFilter;
    private final LogoutSuccessHandler logoutSuccessHandler;

    private final HttpCookieOAuth2AuthorizationRequestRepository httpCookieOAuth2AuthorizationRequestRepository;

    public SecurityConfig(
        @Qualifier("customAuthenticationEntryPoint") AuthenticationEntryPoint authEntryPoint,
        @Qualifier("customOAuth2AuthenticationFailureHandler") AuthenticationFailureHandler authFailureHandler,
        @Qualifier("customOAuth2AuthenticationSuccessHandler") AuthenticationSuccessHandler authenticationSuccessHandler,
        @Qualifier("tokenAuthenticationProcessingFilter") TokenAuthenticationProcessingFilter tokenAuthenticationProcessingFilter,
        @Qualifier("customLogoutHandler") LogoutSuccessHandler logoutSuccessHandler,
        @Autowired HttpCookieOAuth2AuthorizationRequestRepository httpCookieOAuth2AuthorizationRequestRepository
    ) {
        this.authEntryPoint = authEntryPoint;
        this.authFailureHandler = authFailureHandler;
        this.authSuccessHandler = authenticationSuccessHandler;
        this.tokenAuthenticationProcessingFilter = tokenAuthenticationProcessingFilter;
        this.logoutSuccessHandler = logoutSuccessHandler;
        this.httpCookieOAuth2AuthorizationRequestRepository = httpCookieOAuth2AuthorizationRequestRepository;
    }

    @Bean
    public SecurityFilterChain securityFilterChain(
        HttpSecurity httpSecurity,
        CustomOAuth2UserService customOAuth2UserService
    ) throws Exception {
        return httpSecurity
            .authorizeHttpRequests(auth -> auth
                .requestMatchers(
                    PathRequest.toStaticResources().atCommonLocations()
                ).permitAll()
                .requestMatchers(
                    HttpMethod.GET,
                    "/", "/oauth2/**"
                ).permitAll()
                .requestMatchers(
                    HttpMethod.POST,
                    "/auth/token/refresh", "/oauth2/**", "/upload"
                ).permitAll()
                // ws 연결을 위한 /ws url 모두 허용
                .requestMatchers(
                    "/ws", "/test2"
                ).permitAll()
                .anyRequest().authenticated()
            )
            .sessionManagement(session -> session
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            )
            .csrf(
                csrf -> csrf
                    .ignoringRequestMatchers("/api/**")
                    .disable()
            )
            .cors(Customizer.withDefaults())
            .addFilterBefore(
                tokenAuthenticationProcessingFilter,
                UsernamePasswordAuthenticationFilter.class
            )
            .formLogin(AbstractHttpConfigurer::disable)
            .oauth2Login(oAuth -> oAuth
                .authorizationEndpoint(authorization -> authorization
                    .baseUri("/oauth2/authorize")
                    .authorizationRequestRepository(httpCookieOAuth2AuthorizationRequestRepository)
                )
                .redirectionEndpoint(redirect -> redirect
                    .baseUri("/oauth2/callback/*")
                )
                .userInfoEndpoint(userInfo -> userInfo
                    .userService(customOAuth2UserService)
                )
                .failureHandler(authFailureHandler)
                .successHandler(authSuccessHandler)
            )
            .logout(logout -> logout
                .logoutUrl("/auth/logout")
                .logoutSuccessHandler(logoutSuccessHandler)
                .deleteCookies("JSESSIONID")
            )
            .exceptionHandling(hc -> hc
                .authenticationEntryPoint(authEntryPoint)
            )
            .build();
    }
}
