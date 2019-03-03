package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;

public class BreastCancerTest {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 31, outputLayer = 1, trainingIterations = 500;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        System.out.println("running code...");
        int hiddenLayerLower = 1;
        int hiddenLayerUpper = 300;
        int count = 0;
        double[] accuracyArr = new double[hiddenLayerUpper - hiddenLayerLower];
        double maxHidden = 0;
        double maxCorrect = 0;
        String maxResult = "";
        int count2 = 0;
        // int trainingIterations1 = Integer.parseInt(args[3]);

        int hiddenLayer = 10;

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        System.out.println("NeuralNetworkOptimizationProblem instance created");

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        double temp = 6;
        double cool = .50;
        oa[1] = new SimulatedAnnealing(temp, cool, nnop[1]);
        int populationSize = Integer.parseInt(args[0]);
        int toMate = Integer.parseInt(args[1]);
        int toMutate = Integer.parseInt(args[2]);
        oa[2] = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, nnop[2]);

        System.out.println("StandardGeneticAlgorithm instance created");


        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            double predicted, actual;
            start = System.nanoTime();

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            predicted = 0;
            actual = 0;
            start = System.nanoTime();
//
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            try {
                FileWriter rhc_writer = new FileWriter("rhc.csv", true);
                FileWriter sa_writer = new FileWriter("sa.csv", true);
                FileWriter ga_writer = new FileWriter("ga.csv", true);

                if (i == 0) {
                    rhc_writer.append("" + correct/(correct+incorrect)*100);
                    rhc_writer.append(",");
                    rhc_writer.append("" + trainingIterations);
                    rhc_writer.append(",");
                    rhc_writer.append("" + 20);
                    rhc_writer.append(",");
                    rhc_writer.append("" + trainingTime);
                    rhc_writer.append(",");
                    rhc_writer.append("" + testingTime);
                    rhc_writer.append("\n");
                } else if (i == 1) {
                    sa_writer.append("" + correct/(correct+incorrect)*100);
                    sa_writer.append(",");
                    sa_writer.append("" + cool);
                    sa_writer.append(",");
                    sa_writer.append("" + temp);
                    sa_writer.append(",");
                    sa_writer.append("" + trainingIterations);
                    sa_writer.append(",");
                    sa_writer.append("" + trainingTime);
                    sa_writer.append(",");
                    sa_writer.append("" + testingTime);
                    sa_writer.append("\n");
                } else {
                    ga_writer.append("" + correct/(correct+incorrect)*100);
                    ga_writer.append(",");
                    ga_writer.append("" + populationSize);
                    ga_writer.append(",");
                    ga_writer.append("" + toMate);
                    ga_writer.append(",");
                    ga_writer.append("" + toMutate);
                    ga_writer.append(",");
                    ga_writer.append("" + trainingIterations);
                    ga_writer.append(",");
                    ga_writer.append("" + trainingTime);
                    ga_writer.append(",");
                    ga_writer.append("" + testingTime);
                    ga_writer.append("\n");
                }

                rhc_writer.close();
                sa_writer.close();
                ga_writer.close();
            } catch (IOException e) {
                System.out.println("Not Written");
                e.printStackTrace();
            }

        }

        System.out.println("results" + results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        // System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

//            double error = 0;
//            for(int j = 0; j < instances.length; j++) {
//                network.setInputValues(instances[j].getData());
//                network.run();
//
//                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
//                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
//                error += measure.value(output, example);
//            }

            // System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[569][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/breast_cancer.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[31]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 31; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
