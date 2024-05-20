package org.capstone.maru.dto.request;

public record RecommendRequest(
    String user_id,
    String card_type,
    String want_to_find
) {

}
