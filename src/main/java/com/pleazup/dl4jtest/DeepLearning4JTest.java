package com.pleazup.dl4jtest;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by marius on 2/10/17.
 */
public class DeepLearning4JTest {

    private static Logger log = LoggerFactory.getLogger(DeepLearning4JTest.class);

    public static void main(String[] args) {
        File gModel = new File("/media/data/repo/deeplearning4j/GoogleNews-vectors-negative300.bin.gz");
        Word2Vec vec = WordVectorSerializer.readWord2VecModel(gModel);
        Collection<String> list = vec.wordsNearest(Arrays.asList("Japan", "Beijing"), Arrays.asList("China"), 10);
        System.out.println(list);
    }
}
