package org.pcj.fft;

/**
 * @author Łukasz Górski <lgorski@mat.umk.pl>
 */
public class Utilities {

    public static int number_of_bits(long num) {
        int n = 0;
        while (num > 0) {
            n++;
            num >>>= 1;
        }
        return n;
    }
}
