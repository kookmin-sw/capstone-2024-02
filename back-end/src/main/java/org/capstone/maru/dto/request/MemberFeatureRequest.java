package org.capstone.maru.dto.request;

import java.util.List;

public record MemberFeatureRequest(
    String location,
    List<String> features
) {

}
