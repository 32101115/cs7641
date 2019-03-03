package opt.test;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import opt.example.TravelingSalesmanEvaluationFunction;
import opt.example.TravelingSalesmanRouteEvaluationFunction;
import opt.NeighborFunction;
import opt.ga.MutationFunction;
import opt.ga.CrossoverFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.SwapNeighbor;
import opt.ga.SwapMutation;
import opt.example.TravelingSalesmanCrossOver;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.StandardGeneticAlgorithm;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * A test using the 4 Peaks evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {

    public static void main(String[] args) {
        if(args.length < 2)
        {
            System.out.println("Provide a input size and repeat count");
            System.exit(0);
        }
        int N = Integer.parseInt(args[0]);
        if(N < 0)
        {
            System.out.println(" N cannot be negaitve.");
            System.exit(0);
        }
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        int iterations = Integer.parseInt(args[1]);
        int[] ranges = new int[N];

        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        Distribution df = new DiscreteDependencyTree(.1, ranges);
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
                FileWriter ff_rhc = new FileWriter("ts_rhc.csv", true);

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
                FileWriter ff_rhc = new FileWriter("ts_sa.csv", true);

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
                FileWriter ff_rhc = new FileWriter("ts_ga.csv", true);

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
       //          FileWriter mmc = new FileWriter("ts_mimic.csv", true);

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
