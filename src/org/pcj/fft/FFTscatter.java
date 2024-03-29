package org.pcj.fft;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.pcj.PCJ;
import org.pcj.RegisterStorage;
import org.pcj.StartPoint;
import org.pcj.Storage;

@RegisterStorage(FFTscatter.Shareable.class)
public class FFTscatter implements StartPoint {

    public final static boolean DO_PRINT = false;

    @Storage(FFTscatter.class)
    enum Shareable {
        dummy, blocks
    }

    private boolean dummy;
    private double[][] blocks = new double[PCJ.threadCount()][];


    private long n, two_n, world_size, world_logsize;
    private long local_size, local_logsize;
    private long tstart, tend, rate, mystart, rank;
    private double tdelta, gflops, tsec;
    private long t_all = 0L;
    private long seed = 0L;

    public static void main(String[] args) throws IOException {
        PCJ.executionBuilder(FFTscatter.class)
                .addNodes(new File(args.length > 0 ? args[0] : "nodes.txt"))
                .start();
    }

    public void print_array(double[] arr, String name) {
        //       System.err.println(name);
        String out = "";
        for (int z = 0; z < arr.length / 2; z++) {
            out += "#" + PCJ.myId() + " " + name + " = " + arr[2 * z] + " " + arr[2 * z + 1] + "i\n";
        }
        System.out.println(out);
    }

    @Override
    public void main() throws Throwable {
        world_size = PCJ.threadCount();
        world_logsize = number_of_bits(world_size) - 1L;
        rank = PCJ.myId();
        BufferedReader br = new BufferedReader(new FileReader("size.txt"));
        int global_logsize = Integer.parseInt(br.readLine());
        local_logsize = global_logsize - world_logsize;

        local_size = 1L << local_logsize;

        bitinit();

        mystart = rank * local_size;

        long time = Long.MAX_VALUE;
        long t_all_old = Long.MAX_VALUE;
        for (int i = 0; i < 5; i++) {
            PCJ.barrier();
            fft_init(local_size, mystart, rank);
            tstart = System.nanoTime();
            fft_inner(local_size, 1L);

            tend = System.nanoTime();
            PCJ.barrier();
            if (tend - tstart < time) {
                time = tend - tstart;
                t_all_old = t_all;
            }
        }

        if (DO_PRINT) {
            if (PCJ.myId() != 0L) {
                PCJ.waitFor(Shareable.dummy);
            }

            for (int i = 0; i < c.length / 2; i++) {
                System.out.println("#" + PCJ.myId() + " " + c[i * 2] + " + " + c[i * 2 + 1] + "i");
            }

            if (PCJ.myId() != PCJ.threadCount() - 1) {
                PCJ.put(true, PCJ.myId() + 1, Shareable.dummy);
            }
        }
        PCJ.barrier();

        fft_verif(world_size, local_size, mystart);

        PCJ.barrier();

        if (rank == 0) {
            n = local_logsize + world_logsize;
            two_n = 1L << n;
            tsec = (time) * 1e-9;
            System.out.println("Elapsed time: " + tsec);
            gflops = ((5.0d * n * two_n) / tsec) * 1e-9;
            System.out.println("Num PEs " + +world_size + " Local size: " + local_size + " GFlops = " + gflops);
        }
        System.err.flush();
        PCJ.barrier();
    }

    private double[] c;
    private double[] spare;
    private double[] twiddles;

    void fft_init(long n_local_size, long local_start, long rank) {
        long i;
        twiddles = new double[(int) (n_local_size)];
        c = new double[(int) (2L * n_local_size)];
        spare = new double[(int) (2L * n_local_size)];
        t_all = 0L;

        Random random = new Random();
        seed = random.nextLong();

        initialize_data_array(n_local_size, local_start, rank, c);
    }

    private long mask64, mask32, mask16, mask8, mask4, mask2, mask1;

    public void bitinit() {
        mask64 = ~0L;
        mask32 = mask64 << 32L;
        mask16 = (mask32 >>> 16L) ^ mask32;
        mask8 = (mask16 >>> 8L) ^ mask16;
        mask4 = (mask8 >>> 4L) ^ mask8;
        mask2 = (mask4 >>> 2L) ^ mask4;
        mask1 = (mask2 >>> 1L) ^ mask2;
    }

    public long interchange(long ival, long imask, long ishift) {
        return ((ival & imask) >>> ishift) | ((ival & ~imask) << ishift);
    }

    public long i_bitreverse(long i, long n) {
        long itmp;
        long out;
        itmp = interchange(i, mask32, 32L);
        itmp = interchange(itmp, mask16, 16L);
        itmp = interchange(itmp, mask8, 8L);
        itmp = interchange(itmp, mask4, 4L);
        itmp = interchange(itmp, mask2, 2L);
        itmp = interchange(itmp, mask1, 1L);

        if (n - 64L < 0L) {
            out = itmp >>> -(n - 64L);
        } else {
            out = itmp << (n - 64L);
        }
        return out;
    }

    private void initialize_data_array(long n_local_size, long local_start, long rank, double[] buffer) {
        long i, j;
        int n;
        //int seed = 314159265;
        double h, h2;
        Random random = new Random(seed);
        for (i = 0L; i < buffer.length / 2L; i++) {
            //                  buffer[(int) (2L * i)] = PCJ.myId();
            //                  buffer[(int) (2L * i) + 1L] = -PCJ.myId();
            buffer[(int) (2L * i)] = random.nextDouble();
            buffer[(int) (2L * i + 1L)] = random.nextDouble();
        }
    }

    private void fft_inner(long n_local_size, long direction) {
        long world_size, n_world_size;
        long rank, lo, hi, lstride, i, j, k;
        long levels, l, loc_comm, m, m2;
        double cer, cei, crr, cri, clr, cli; //complex
        double two_pi, angle_base;

        rank = PCJ.myId();
        world_size = PCJ.threadCount();
        n_world_size = world_size * n_local_size;
        two_pi = 2.0d * Math.PI * direction;

        levels = number_of_bits(n_world_size) - 1L;
        loc_comm = number_of_bits(n_local_size);
        long tt = 0L;

        long t1 = System.nanoTime();
        permute(c, n_world_size, spare);
        long t2 = System.nanoTime();

        tt += t2 - t1;

        for (i = 0L; i < n_local_size / 2L; i++) {

            double x = 0.0d;
            double y = ((i) * ((-two_pi) / n_local_size));
            spare[(int) (2L * i)] = Math.exp(x) * Math.cos(y);
            spare[(int) (2L * i + 1L)] = Math.exp(x) * Math.sin(y);
        }

        //phase 1L computation
        for (l = 1L; l < loc_comm; l++) {
            m = 1L << l;
            m2 = m >>> 1L;
            lstride = n_local_size >>> l;
            for (int zz = 0, aa = 0; zz < m2; zz++, aa += lstride) {
                twiddles[2 * zz] = spare[2 * aa];
                twiddles[2 * zz + 1] = spare[2 * aa + 1];
            }
            for (k = 0L; k < n_local_size; k += m) {
                for (j = k; j < k + m2; j++) {
                    cer = twiddles[(int) (2L * (j - k))];
                    cei = twiddles[(int) (2L * (j - k) + 1L)];

                    double wr = c[(int) (2L * (j + m2))];
                    double wi = c[(int) (2L * (j + m2) + 1L)];

                    crr = cer * wr - cei * wi;
                    cri = cer * wi + cei * wr;
                    clr = c[(int) (2L * j)];
                    cli = c[(int) (2L * j + 1L)];
                    c[(int) (2L * j)] = clr + crr;
                    c[(int) (2L * j + 1L)] = cli + cri;
                    c[(int) (2L * (j + m2))] = clr - crr;
                    c[(int) (2L * (j + m2) + 1L)] = cli - cri;
                }
            }
        }

        t1 = System.nanoTime();
        transpose(c, n_world_size, world_size, spare, true);
        t2 = System.nanoTime();

        tt += t2 - t1;

        for (l = loc_comm; l <= levels; l++) {
            m = (1L << l) / world_size;
            m2 = m >>> 1L;
            angle_base = (-two_pi) / (1L << l);
            for (k = 0L; k < n_local_size; k += m) {
                for (j = k; j < k + m2; j++) {
                    double x = 0.0d;
                    double y = (((j - k) * world_size + rank) * angle_base);
                    cer = Math.exp(x) * Math.cos(y);
                    cei = Math.exp(x) * Math.sin(y);

                    double wr = c[(int) (2L * (j + m2))];
                    double wi = c[(int) (2L * (j + m2) + 1L)];

                    crr = cer * wr - cei * wi;
                    cri = cer * wi + cei * wr;

                    clr = c[(int) (2L * j)];
                    cli = c[(int) ((2L * j) + 1L)];

                    c[(int) (2L * j)] = clr + crr;
                    c[(int) ((2L * j) + 1L)] = cli + cri;
                    c[(int) (2L * (j + m2))] = clr - crr;
                    c[(int) (2L * (j + m2) + 1L)] = cli - cri;
                }
            }
        }
        t1 = System.nanoTime();
        transpose(c, n_world_size, n_local_size / world_size, spare, false);
        t2 = System.nanoTime();
        tt += t2 - t1;
        //      System.out.println("#" + PCJ.myId() + " communication: " + tt * 1e-9 + " s");
    }

    private void permute(double[] c, long n, double[] scratch) {
        long world_size, local_n, block_size;

        world_size = PCJ.threadCount();
        local_n = n / world_size;
        block_size = local_n / world_size;
        packf(c, scratch, local_n, world_size, 32L, 1024L, 0L, true);

        alltoall(scratch, c, block_size);
        System.arraycopy(c, 0, scratch, 0, c.length);
        permute_locally(c, scratch, local_n, 1L << 22L);

    }

    private void packf(double[] input, double[] output, long n, long p, long n_b, long cp_b, long npadding, boolean do_bitreverse) {
        long[] pe_bufstart = new long[(int) p];
        long pe_buflen, buf, p_bits, ii, i, jj, j, ooffset, ioffset, p_b;

        p_b = cp_b;

        p_bits = number_of_bits(p - 1L);
        pe_buflen = n / p + npadding;

        if (do_bitreverse) {
            for (i = 0L; i < p; i++) {
                pe_bufstart[(int) i] = i_bitreverse(i, p_bits) * pe_buflen;
            }
        } else {
            for (i = 0L; i < p; i++) {
                pe_bufstart[(int) i] = i * pe_buflen;
            }
        }

        if (p_b > p) {
            p_b = p;
        }

        for (jj = 0L; jj < p; jj += p_b) {
            ooffset = 0L;
            for (ii = 0L; ii < n; ii += (p * n_b)) {
                for (j = jj; j < jj + p_b; j++) {
                    buf = pe_bufstart[(int) j];
                    ioffset = ooffset;
                    for (i = ii; i <= Math.min(ii + p * (n_b - 1L), n - 1L); i += p) {
                        output[(int) (2L * (buf + ioffset))] = input[(int) (2L * (i + j))];
                        output[(int) (2L * (buf + ioffset) + 1L)] = input[(int) (2L * (i + j) + 1L)];
                        ioffset++;
                    }
                }
                ooffset += n_b;
            }
        }
    }

    void alltoall(double[] source, double[] dest, long blockSize) {
        t_all = System.nanoTime();
        allToAllPerform(source, dest, blockSize);
        t_all = System.nanoTime() - t_all;
    }
    //intermediate buffer


    /**
     * @param source    Data to be sent to other threads - will be put into the
     *                  "blocks" shared array
     * @param blockSize
     */
    void allToAllPerform(double[] source, double[] dest, long blockSize) {
// version 1:
//        Map<Integer, double[]> toSendMap
//                = IntStream.range(0, PCJ.threadCount())
//                          .collect(HashMap::new,
//                                  (m, i) -> {
//                                      double[] toScatter = new double[(int) (2 * blockSize)];
//                                      System.arraycopy(source, (int) (2 * i * blockSize), toScatter, 0, (int) (2 * blockSize));
//                                      m.put(i, toScatter);
//                                  },
//                                  Map::putAll);
// version 2:
//        Map<Integer, double[]> toSendMap = IntStream.range(0, PCJ.threadCount())
//                .boxed()
//                .collect(Collectors.toMap(
//                        Function.identity(),
//                        i -> {
//                            double[] toScatter = new double[(int) (2 * blockSize)];
//                            System.arraycopy(source, (int) (2 * i * blockSize), toScatter, 0, (int) (2 * blockSize));
//                            return toScatter;
//                        }));
        Map<Integer, double[]> toSendMap = IntStream.range(0, PCJ.threadCount())
                   .boxed()
                   .collect(Collectors.toMap(
                           Function.identity(),
                           i -> Arrays.copyOfRange(source, (int) (2 * i * blockSize), (int) (2 * (i + 1) * blockSize))));
        PCJ.scatter(toSendMap, Shareable.blocks, PCJ.myId());

        PCJ.waitFor(Shareable.blocks, PCJ.threadCount());
        for (int i = 0; i < blocks.length; ++i) {
            System.arraycopy(blocks[i], 0, dest, (int) (i * 2 * blockSize), (int) (2 * blockSize));
            blocks[i] = null;
        }
        PCJ.barrier();
    }


    private void transpose(double[] c, long n, long n_transpose, double[] scratch, boolean forward) {
        long world_size = PCJ.threadCount();
        long local_n = n / world_size;
        long block_size = local_n / world_size;

        if (forward) {
            packf(c, scratch, local_n, n_transpose, 32L, 1024L, 0L, false);
            alltoall(scratch, c, block_size);
            System.arraycopy(c, 0, scratch, 0, c.length);
        } else {
            alltoall(c, scratch, block_size);
            System.arraycopy(scratch, 0, c, 0, scratch.length);
            packf(scratch, c, local_n, n_transpose, 32L, 1024L, 0L, false);
        }
    }

    private void fft_verif(long world_size, long n_local_size, long local_start) {
        long i, j, rank, mei;
        double norm, error, max_error, residue, logm;
        final double epsilon = 1.1e-16;
        final double max_residue = 16.0d;

        //perform inverse fft
        for (i = 0L; i < n_local_size; i++) {

            double wr = world_size * n_local_size;
            double wi = 0.0d;

            double mod = ((wr != 0.0 || wi != 0.0d) ? Math.sqrt(wr * wr + wi * wi) : 0.0d);
            double den = Math.pow(mod, 2.0d);

            double cr = c[(int) (2L * i)];
            double ci = c[(int) ((2L * i) + 1L)];

            cr = (cr * wr + ci * wi) / den;
            ci = (ci * wr - cr * wi) / den;

            c[(int) (2L * i)] = cr;
            c[(int) (2L * i + 1L)] = ci;

        }
        fft_inner(n_local_size, -1L);

        rank = PCJ.myId();
        initialize_data_array(n_local_size, local_start, rank, spare);

        max_error = -1.0d;
        mei = -1L;
        for (i = 0L; i < n_local_size; i++) {

            double er = c[(int) (2L * i)] - spare[(int) (2L * i)];
            double ei = c[(int) (2L * i + 1L)] - spare[(int) (2L * i + 1L)];
            error = Math.abs(er * er + ei * ei);
            if (error > max_error) {
                mei = i;
            }
            max_error = Math.max(max_error, error);
        }

        logm = number_of_bits(world_size) - 1L + number_of_bits(n_local_size) - 1L;
        residue = (max_error / epsilon) / logm;
        if (residue < max_residue && rank == 0L) {
            System.out.println("Verification successful");
        } else {
            if (residue >= max_residue) {
                System.out.println("Verification failed (residue = " + residue + ")");
                System.out.println("   Max error: " + max_error);
                System.out.println("   In: (" + c[(int) mei] + "); Out: (" + spare[(int) mei] + ")");
            }
        }
    }

    private void permute_locally(double[] dest, double[] src, long n, long cn_b) {
        long i, j, n_bits, n_b;

        n_b = cn_b;

        n_bits = number_of_bits(n - 1L);

        if (n_b > n) {
            n_b = n;
        }

        for (j = 0L; j <= n_b; j++) {
            for (i = j; i < n; i += n_b) {
                dest[(int) (2L * i_bitreverse(i, n_bits))] = src[(int) (2L * i)];
                dest[(int) (2L * i_bitreverse(i, n_bits) + 1L)] = src[(int) (2L * i + 1L)];
            }
        }
    }

    private static int number_of_bits(long num) {
        int n = 0;
        while (num > 0) {
            n++;
            num >>>= 1;
        }
        return n;
    }
}


/* Code is based on CAF 2.0 implementation from Rice University. The license terms of source code this version is based on is as follows:

Copyright (c) 2009-2011 Rice University.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of Rice University (RICE) nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

This software is provided by RICE and contributors "as is" and any
express or implied warranties, including, but not limited to, the
implied warranties of merchantability and fitness for a particular
purpose are disclaimed. In no event shall RICE or contributors be
liable for any direct, indirect, incidental, special, exemplary, or
consequential damages (including, but not limited to, procurement of
substitute goods or services; loss of use, data, or profits; or
business interruption) however caused and on any theory of liability,
whether in contract, strict liability, or tort (including negligence
or otherwise) arising in any way out of the use of this software, even
if advised of the possibility of such damage.


*/
