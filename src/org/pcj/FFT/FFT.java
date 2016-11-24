/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.aaaa
 */
package org.pcj.FFT;

import java.io.BufferedReader;
import java.io.FileReader;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.pcj.FutureObject;
import org.pcj.PCJ;
import org.pcj.Shared;
import org.pcj.StartPoint;
import org.pcj.Storage;

public class FFT extends Storage implements StartPoint {

    public final static boolean DO_PRINT = false;
    @Shared
    public boolean dummy = false;
    long n, two_n, world_size, world_logsize;
    long local_size, local_logsize;
    long tstart, tend, rate, mystart, rank;
    double tdelta, gflops, tsec;
    long t_all = 0;
    long seed = 0;

    public static void main(String[] args) {
        String nodesFileName = "nodes.txt";
        if (args.length > 0) {
            nodesFileName = args[0];
        }
        PCJ.start(FFT.class, FFT.class, nodesFileName);
    }

    public void print_array(double[] arr, String name) {
        //       System.err.println(name);
        String out = "";
        for (int z = 0; z < arr.length / 2; z++) {
            out += "#" + PCJ.myId() + " " + name + " = " + arr[2 * z] + " " + arr[2 * z + 1] + "i\n";
        }
        PCJ.log(out);
    }

    @Override
    public void main() throws Throwable {
        world_size = PCJ.threadCount();
        world_logsize = Utilities.number_of_bits(world_size) - 1;
        rank = PCJ.myId();
        BufferedReader br = new BufferedReader(new FileReader("size.txt"));
        int global_logsize = Integer.parseInt(br.readLine());
        local_logsize = global_logsize - world_logsize;

        local_size = 1 << local_logsize;

        bitinit();

        mystart = rank * local_size;

        long time = Long.MAX_VALUE;
        long t_all_old = Long.MAX_VALUE;
        for (int i = 0; i < 5; i++) {
            PCJ.barrier();
            fft_init(local_size, mystart, rank);
            tstart = System.currentTimeMillis();
            fft_inner(local_size, 1L);

            tend = System.currentTimeMillis();
            PCJ.barrier();
            if (tend - tstart < time) {
                time = tend - tstart;
                t_all_old = t_all;
            }
        }

        if (DO_PRINT) {
            if (PCJ.myId() != 0) {
                PCJ.waitFor("dummy");
            }

            for (int i = 0; i < c.length / 2; i++) {
                PCJ.log("#" + PCJ.myId() + " " + c[i * 2] + " + " + c[i * 2 + 1] + "i");
            }

            if (PCJ.myId() != PCJ.threadCount() - 1) {
                PCJ.put(PCJ.myId() + 1, "dummy", true);
            }
        }
        PCJ.barrier();
         
         fft_verif(world_size, local_size, mystart);
         
        PCJ.barrier();

        if (rank == 0) {
            n = local_logsize + world_logsize;
            two_n = 1 << n;
            tsec = (time) * 1e-3;
            System.out.println("Elapsed time: " + tsec);
            gflops = ((5.0 * n * two_n) / tsec) * 1e-9;
            System.out.println("Num PEs " + +world_size + " Local size: " + local_size + " GFlops = " + gflops);
        }
        //       PCJ.log("#" + PCJ.myId() + " Alltoall time = " + t_all_old * 1e-3);
        System.err.flush();
        PCJ.barrier();
    }
    double[] c;
    double[] spare;
    double[] twiddles;

    void fft_init(long n_local_size, long local_start, long rank) {
        long i;
        twiddles = new double[(int) (n_local_size)];
        c = new double[(int) (2 * n_local_size)];
        spare = new double[(int) (2 * n_local_size)];
        t_all = 0;

        Random random = new Random();
        seed = random.nextLong();

        initialize_data_array(n_local_size, local_start, rank, c);
    }
    long mask64, mask32, mask16, mask8, mask4, mask2, mask1;

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

        if (n - 64L < 0) {
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
        for (i = 0; i < buffer.length / 2; i++) {
    //                             buffer[(int) (2 * i)] = PCJ.myId();
      //                  buffer[(int) (2 * i) + 1] = -PCJ.myId();
            buffer[(int) (2 * i)] = random.nextDouble();
            buffer[(int) (2 * i) + 1] = random.nextDouble();
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

        levels = Utilities.number_of_bits(n_world_size) - 1;
        loc_comm = Utilities.number_of_bits(n_local_size);
        long tt = 0;

        long t1 = System.currentTimeMillis();
        permute(c, n_world_size, spare);
        long t2 = System.currentTimeMillis();

        tt += t2 - t1;

        for (i = 0; i < n_local_size / 2; i++) {

            double x = 0.0f;
            double y = (double) ((i) * ((-two_pi) / n_local_size));
            spare[(int) (2 * i)] = (double) (Math.exp(x) * Math.cos(y));
            spare[(int) (2 * i + 1)] = (double) (Math.exp(x) * Math.sin(y));
        }

        //phase 1 computation
        for (l = 1; l < loc_comm; l++) {
            m = 1 << l;
            m2 = m >>> 1;
            lstride = n_local_size >>> l;
            for (int zz = 0, aa = 0; zz < m2; zz++, aa += lstride) {
                twiddles[2 * zz] = spare[2 * aa];
                twiddles[2 * zz + 1] = spare[2 * aa + 1];
            }
            for (k = 0; k < n_local_size; k += m) {
                for (j = k; j < k + m2; j++) {
                    cer = twiddles[(int) (2 * (j - k))];
                    cei = twiddles[(int) (2 * (j - k) + 1)];

                    double wr = c[(int) (2 * (j + m2))];
                    double wi = c[(int) (2 * (j + m2) + 1)];

                    crr = cer * wr - cei * wi;
                    cri = cer * wi + cei * wr;
                    clr = c[(int) (2 * j)];
                    cli = c[(int) (2 * j + 1)];
                    c[(int) (2 * j)] = clr + crr;
                    c[(int) (2 * j + 1)] = cli + cri;
                    c[(int) (2 * (j + m2))] = clr - crr;
                    c[(int) (2 * (j + m2) + 1)] = cli - cri;
                }
            }
        }

        t1 = System.currentTimeMillis();
        transpose(c, n_world_size, world_size, spare, true);
        t2 = System.currentTimeMillis();

        tt += t2 - t1;

        for (l = loc_comm; l <= levels; l++) {
            m = (1 << l) / world_size;
            m2 = m >>> 1;
            angle_base = (-two_pi) / (1 << l);
            for (k = 0; k < n_local_size; k += m) {
                for (j = k; j < k + m2; j++) {
                    double x = 0.0f;
                    double y = (double) (((j - k) * world_size + rank) * angle_base);
                    cer = (double) (Math.exp(x) * Math.cos(y));
                    cei = (double) (Math.exp(x) * Math.sin(y));

                    double wr = c[(int) (2 * (j + m2))];
                    double wi = c[(int) (2 * (j + m2) + 1)];

                    crr = cer * wr - cei * wi;
                    cri = cer * wi + cei * wr;

                    clr = c[(int) (2 * j)];
                    cli = c[(int) ((2 * j) + 1)];

                    c[(int) (2 * j)] = clr + crr;
                    c[(int) ((2 * j) + 1)] = cli + cri;
                    c[(int) (2 * (j + m2))] = clr - crr;
                    c[(int) (2 * (j + m2) + 1)] = cli - cri;
                }
            }
        }
        t1 = System.currentTimeMillis();
        transpose(c, n_world_size, n_local_size / world_size, spare, false);
        t2 = System.currentTimeMillis();
        tt += t2 - t1;
        //      System.out.println("#" + PCJ.myId() + " communication: " + tt * 1e-3 + " s");
    }

    private void permute(double[] c, long n, double[] scratch) {
        long world_size, local_n, block_size;

        world_size = PCJ.threadCount();
        local_n = n / world_size;
        block_size = local_n / world_size;
        packf(c, scratch, local_n, world_size, 32L, 1024L, 0L, true);

        alltoall(scratch, c, block_size);
        System.arraycopy(c, 0, scratch, 0, c.length);
        permute_locally(c, scratch, local_n, 1L << 22);

    }

    private void packf(double[] input, double[] output, long n, long p, long n_b, long cp_b, long npadding, boolean do_bitreverse) {
        long[] pe_bufstart = new long[(int) p];
        long pe_buflen, buf, p_bits, ii, i, jj, j, ooffset, ioffset, p_b;

        p_b = cp_b;

        p_bits = Utilities.number_of_bits(p - 1);
        pe_buflen = n / p + npadding;

        if (do_bitreverse) {
            for (i = 0; i < p; i++) {
                pe_bufstart[(int) i] = i_bitreverse(i, p_bits) * pe_buflen;
            }
        } else {
            for (i = 0; i < p; i++) {
                pe_bufstart[(int) i] = i * pe_buflen;
            }
        }

        if (p_b > p) {
            p_b = p;
        }

        for (jj = 0; jj < p; jj += p_b) {
            ooffset = 0;
            for (ii = 0; ii < n; ii += (p * n_b)) {
                for (j = jj; j < jj + p_b; j++) {
                    buf = pe_bufstart[(int) j];
                    ioffset = ooffset;
                    for (i = ii; i <= Math.min(ii + p * (n_b - 1), n - 1); i += p) {
                        output[(int) (2 * (buf + ioffset))] = input[(int) (2 * (i + j))];
                        output[(int) (2 * (buf + ioffset) + 1)] = input[(int) (2 * (i + j) + 1)];
                        ioffset++;
                    }
                }
                ooffset += n_b;
            }
        }
    }

    void alltoall(double[] source, double[] dest, long blockSize) {
        t_all = System.currentTimeMillis();
        allToAllPerform(source, dest, blockSize);
        t_all = System.currentTimeMillis() - t_all;
    }
    //intermediate buffer
    @Shared
    double[][] blocks;

    /**
     *
     * @param source Data to be sent to other threads - will be put into the
     * "blocks" shared array
     * @param blockSize
     */
    void allToAllPerform(double[] source, double[] dest, long blockSize) {
        
        prepareAllToAll(blockSize, source);
        PCJ.barrier();
       // allToAllNonBlocking(dest, blockSize);  
             allToAllBlocking(dest, blockSize);
        System.arraycopy(source, PCJ.myId() * (int) (2 * blockSize), dest, PCJ.myId() * (int) (2 * blockSize), (int) (2 * blockSize));
        PCJ.barrier();
       /* System.arraycopy(source, PCJ.myId() * (int) (2 * blockSize), dest, PCJ.myId() * (int) (2 * blockSize), (int) (2 * blockSize));
                if (PCJ.myId() == 0 || PCJ.myId() == 1) {
            System.out.println(PCJ.myId() + " we" + Arrays.toString(source));
            System.out.println(PCJ.myId() + " wy" + Arrays.toString(dest));
        } */    
        
      //  alltoallHypercube(source, dest, blockSize);
      /*  System.arraycopy(source, PCJ.myId() * (int) (2 * blockSize), dest, PCJ.myId() * (int) (2 * blockSize), (int) (2 * blockSize));
                if (PCJ.myId() == 0 || PCJ.myId() == 1) {
            System.out.println(PCJ.myId() + " we" + Arrays.toString(source));
            System.out.println(PCJ.myId() + " wy" + Arrays.toString(dest));
        }*/
    }

    @Shared
    double[][][] blocksHypercube;

    double[][] blocked;

    private void alltoallHypercube(double[] src, double[] dest, long blockSize) {
        PCJ.barrier();
        blocked = new double[PCJ.threadCount()][(int) (2 * blockSize)];
        for (int i = 0; i < blocked.length; i++) {
            System.arraycopy(src, i * 2 * (int) blockSize, blocked[i], 0, (int) (2 * blockSize));
        }
        //all-to-all hypercube personalized communication, per 
        //http://www.sandia.gov/~sjplimp/docs/cluster06.pdf, p. 5.
        int logNumProcs = (int) (Math.log(PCJ.threadCount()) / Math.log(2));
        double[][][] blocksLocal = new double[logNumProcs][][];
        PCJ.putLocal("blocksHypercube", blocksLocal);
        PCJ.barrier();
        int myId = PCJ.myId();

        double[][] toSend = new double[PCJ.threadCount() / 2][];
        for (int dimension = 0; dimension < logNumProcs; dimension++) {

            int partner = (1 << dimension) ^ myId;

            long mask = 1L << dimension;
            int j = 0;
            for (int i = 0; i < PCJ.threadCount(); i++) {
                if (partner < myId) {
                    if ((i & mask) == 0) {
                        toSend[j++] = blocked[i];
                    }
                } else {
                    if ((i & mask) != 0) {
                        toSend[j++] = blocked[i];
                    }
                }
            }

            PCJ.put(partner, "blocksHypercube", toSend, dimension);

            //wait to receive
            double[][] checked = null;
            while (checked == null) {
                j = 0;
                checked = PCJ.getLocal("blocksHypercube", dimension);
                if (checked != null) {

                    PCJ.putLocal("blocksHypercube", null, dimension);
                    for (int i = 0; i < PCJ.threadCount(); i++) {
                        if (partner < myId) {
                            if ((i & mask) == 0) {
                                blocked[i] = checked[j++];
                            }
                        } else {
                            if ((i & mask) != 0) {
                                blocked[i] = checked[j++];
                            }
                        }
                    }
                }
            }
        }

        for (int i = 0; i < blocked.length; i++) {
            System.arraycopy(blocked[i], 0, dest, i * 2 * (int) blockSize, (int) (2 * blockSize));
        }
        PCJ.barrier();
    }

    private void allToAllBlocking(double[] dest, long blockSize) {
        //algorithm inspired from http://www.pgas2013.org.uk/sites/default/files/finalpapers/Day1/H1/3_paper20.pdf
        for (int image = (PCJ.myId() + 1) % PCJ.threadCount(), num = 0; num != PCJ.threadCount() - 1; image = (image + 1) % PCJ.threadCount()) {
            double[] recv = (double[]) PCJ.get(image, "blocks", PCJ.myId());
            System.arraycopy(recv, 0, dest, (int) (image * 2 * blockSize), (int) (2 * blockSize));
            num++;
        }
    }

    private void allToAllNonBlocking(double[] dest, long blockSize) {
        //prepare futures array
        FutureObject<double[]>[] futures = new FutureObject[PCJ.threadCount()];

        //get the data 
        for (int i = 0; i < PCJ.threadCount(); i++) {
            if (i != PCJ.myId()) {
                futures[i] = PCJ.getFutureObject(i, "blocks", PCJ.myId());
            }
        }

        int numReceived = 0;
        while (numReceived != PCJ.threadCount() - 1) {
            for (int i = 0; i < futures.length; i++) {
                if (futures[i] != null && futures[i].isDone()) {
                    double[] recv = futures[i].getObject();
                    System.arraycopy(recv, 0, dest, (int) (i * 2 * blockSize), (int) (2 * blockSize));
                    numReceived++;
                    futures[i] = null;
                }
            }
        }
    }

    private void prepareAllToAll(long blockSize, double[] source) {
        //1) copy information from source to blocks temporary shared array
        blocks = new double[PCJ.threadCount()][];
        for (int i = 0; i < blocks.length; i++) {
            blocks[i] = new double[(int) (2 * blockSize)];
            System.arraycopy(source, 2 * i * (int) blockSize, blocks[i], 0, (int) (2 * blockSize));
        }
        PCJ.putLocal("blocks", blocks);
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
        final double max_residue = 16;

        //perform inverse fft
        for (i = 0; i < n_local_size; i++) {

            double wr = world_size * n_local_size;
            double wi = 0.0f;

            double mod = (double) ((wr != 0 || wi != 0) ? Math.sqrt(wr * wr + wi * wi) : 0.0f);
            double den = (double) Math.pow(mod, 2);

            double cr = c[(int) (2 * i)];
            double ci = c[(int) ((2 * i) + 1)];

            cr = (cr * wr + ci * wi) / den;
            ci = (ci * wr - cr * wi) / den;

            c[(int) (2 * i)] = cr;
            c[(int) (2 * i + 1)] = ci;

        }
        fft_inner(n_local_size, -1L);

        rank = PCJ.myId();
        initialize_data_array(n_local_size, local_start, rank, spare);

        max_error = -1.0;
        mei = -1;
        for (i = 0; i < n_local_size; i++) {

            double er = c[(int) (2 * i)] - spare[(int) (2 * i)];
            double ei = c[(int) (2 * i + 1)] - spare[(int) (2 * i + 1)];
            error = Math.abs(er * er + ei * ei);
            if (error > max_error) {
                mei = i;
            }
            max_error = Math.max(max_error, error);
        }

        logm = Utilities.number_of_bits(world_size) - 1 + Utilities.number_of_bits(n_local_size) - 1;
        residue = (max_error / epsilon) / logm;
        if (residue < max_residue && rank == 0) {
            PCJ.log("Verification successful");
        } else {
            if (residue >= max_residue) {
                PCJ.log("Verification failed (residue = " + residue + ")");
                PCJ.log("   Max error: " + max_error);
                PCJ.log("   In: (" + c[(int) mei] + "); Out: (" + spare[(int) mei] + ")");
            }
        }
    }

    private void permute_locally(double[] dest, double[] src, long n, long cn_b) {
        long i, j, n_bits, n_b;

        n_b = cn_b;

        n_bits = Utilities.number_of_bits(n - 1);

        if (n_b > n) {
            n_b = n;
        }

        for (j = 0; j <= n_b; j++) {
            for (i = j; i < n; i += n_b) {
                dest[(int) (2 * i_bitreverse(i, n_bits))] = src[(int) (2 * i)];
                dest[(int) (2 * i_bitreverse(i, n_bits) + 1)] = src[(int) (2 * i + 1)];
            }
        }
    }
}
