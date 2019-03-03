package opt.test;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.example.FlipFlopEvaluationFunction;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import opt.EvaluationFunction;
import dist.Distribution;
import dist.DiscreteUniformDistribution;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import opt.DiscreteChangeOneNeighbor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

/**
 * A test using the FlipFlop evaluation function
 * @author James Liu
 * @version 1.0
 */
public class FlipFlopTest {

    public static void main(String[] args) {
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
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FlipFlopEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

//        System.out.println("Randomized Hill Climbing\n---------------------------------");
//        for(int i = 0; i < iterations; i++)
//        {
//            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
//            long t = System.nanoTime();
//            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
//            fit.train();
//            try {
//                FileWriter ff_rhc = new FileWriter("ff_rhc.csv", true);
//
//                ff_rhc.append("" + i);
//                ff_rhc.append(",");
//                ff_rhc.append("" + ef.value(rhc.getOptimal()));
//                ff_rhc.append(",");
//                ff_rhc.append("," + (((double)(System.nanoTime() - t))/ 1e9d));
//                ff_rhc.append("\n");
//
//                ff_rhc.close();
//            } catch (IOException e) {
//                System.out.println("Not Written");
//                e.printStackTrace();
//            }
//
//        }
//
//        System.out.println("Simulated Annealing \n---------------------------------");
//        for (int i = 0; i < iterations; i++) {
//            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
//            long t = System.nanoTime();
//            FixedIterationTrainer fit = new FixedIterationTrainer(sa, 200000);
//            fit.train();

//            try {
//                FileWriter ff_rhc = new FileWriter("ff_sa.csv", true);

//                ff_rhc.append("" + i);
//                ff_rhc.append(",");
//                ff_rhc.append("" + ef.value(sa.getOptimal()));
//                ff_rhc.append(",");
//                ff_rhc.append("," + (((double) (System.nanoTime() - t)) / 1e9d));
//                ff_rhc.append("\n");

//                ff_rhc.close();
//            } catch (IOException e) {
//                System.out.println("Not Written");
//                e.printStackTrace();

// //            System.out.println(ef.value(sa.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
//            }

        System.out.println("Genetic Algorithm\n---------------------------------");
        for(int i = 0; i < iterations; i++)
        {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
            long t = System.nanoTime();
            FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
            fit.train();

            try {
                FileWriter ff_rhc = new FileWriter("ff_ga.csv", true);

                ff_rhc.append("" + i);
                ff_rhc.append(",");
                ff_rhc.append("" + ef.value(ga.getOptimal()));
                ff_rhc.append(",");
                ff_rhc.append("," + (((double) (System.nanoTime() - t)) / 1e9d));
                ff_rhc.append("\n");

                ff_rhc.close();
            } catch (IOException e) {
                System.out.println("Not Written");
                e.printStackTrace();

            System.out.println(ef.value(ga.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
        }

       // System.out.println("MIMIC \n---------------------------------");

       // for(int i = 0; i < iterations; i++)
       // {
       //     MIMIC mimic = new MIMIC(200, 100, pop);
       //     long t = System.nanoTime();
       //     FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 1000);
       //     fit.train();

       //     try {
       //          FileWriter mmc = new FileWriter("ff_mimic.csv", true);

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
}
