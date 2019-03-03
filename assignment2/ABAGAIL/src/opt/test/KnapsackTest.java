package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;

import opt.GenericHillClimbingProblem;

import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT =
         MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }

        if (args.length < 2) {
            System.out.println("Provide a input size and repeat counter");
            System.exit(0);
        }
        int N = Integer.parseInt(args[0]);
        if (N < 0) {
            System.out.println(" N cannot be negaitve.");
            System.exit(0);
        }

        int iterations = Integer.parseInt(args[1]);
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);

        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        System.out.println("Randomized Hill Climbing\n---------------------------------");
        for(int i = 0; i < iterations; i++)
        {
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
            fit.train();
            try {
                FileWriter ff_rhc = new FileWriter("kp_rhc.csv", true);

                ff_rhc.append("" + i);
                ff_rhc.append(",");
                ff_rhc.append("" + ef.value(rhc.getOptimal()));
                ff_rhc.append(",");
                ff_rhc.append("," + (((double)(System.nanoTime() - t))/ 1e9d));
                ff_rhc.append("\n");

                ff_rhc.close();
            } catch (IOException e) {
                System.out.println("Not Written");
                e.printStackTrace();
            }
            // System.out.println(ef.value(rhc.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
        }

        System.out.println("Simulated Annealing \n---------------------------------");
        for(int i = 0; i < iterations; i++)
        {
            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            try {
                FileWriter ff_rhc = new FileWriter("kp_sa.csv", true);

                ff_rhc.append("" + i);
                ff_rhc.append(",");
                ff_rhc.append("" + ef.value(sa.getOptimal()));
                ff_rhc.append(",");
                ff_rhc.append("," + (((double)(System.nanoTime() - t))/ 1e9d));
                ff_rhc.append("\n");

                ff_rhc.close();
            } catch (IOException e) {
                System.out.println("Not Written");
                e.printStackTrace();
            }
            // System.out.println(ef.value(sa.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
        }

        System.out.println("Genetic Algorithm\n---------------------------------");
        for(int i = 0; i < iterations; i++)
        {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 10, gap);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            try {
                FileWriter ff_rhc = new FileWriter("kp_ga.csv", true);

                ff_rhc.append("" + i);
                ff_rhc.append(",");
                ff_rhc.append("" + ef.value(ga.getOptimal()));
                ff_rhc.append(",");
                ff_rhc.append("," + (((double)(System.nanoTime() - t))/ 1e9d));
                ff_rhc.append("\n");

                ff_rhc.close();
            } catch (IOException e) {
                System.out.println("Not Written");
                e.printStackTrace();
            }
            // System.out.println(ef.value(ga.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
        }

       // System.out.println("MIMIC \n---------------------------------");

       // for(int i = 0; i < iterations; i++)
       // {
       //     MIMIC mimic = new MIMIC(200, 100, pop);
       //     long t = System.nanoTime();
       //     FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 1000);
       //     fit.train();

       //     try {
       //          FileWriter mmc = new FileWriter("kp_mimic.csv", true);

       //          mmc.append("" + i);
       //          mmc.append(",");
       //          mmc.append("" + ef.value(mimic.getOptimal()));
       //          mmc.append(",");
       //          mmc.append("," + (((double) (System.nanoTime() - t)) / 1e9d));
       //          mmc.append("\n");

       //          mmc.close();
       //      } catch (IOException e) {
       //          System.out.println("Not Written");
       //          e.printStackTrace();
       //     // System.out.println(ef.value(mimic.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
       // }
    }

}
