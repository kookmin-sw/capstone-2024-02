package org.capstone.maru.service;

import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QuerySnapshot;
import com.google.firebase.cloud.FirestoreClient;
import com.google.api.core.ApiFuture;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class FirestoreService {

    public void printAllDocumentsInCollection() {
        Firestore db = FirestoreClient.getFirestore();

        // 컬렉션의 모든 문서 가져오기
        ApiFuture<QuerySnapshot> future = db.collection("recommendation").get();

        try {
            QuerySnapshot querySnapshot = future.get();

            if (!querySnapshot.isEmpty()) {
                querySnapshot.getDocuments().forEach(documentSnapshot -> {
                    log.info("Document ID: {}, Data: {}", documentSnapshot.getId(),
                        documentSnapshot.getData());
                });
            } else {
                log.info("No documents found in the collection.");
            }
        } catch (Exception e) {
            log.error("Error fetching documents: {}", e.getMessage());
        }
    }
}
