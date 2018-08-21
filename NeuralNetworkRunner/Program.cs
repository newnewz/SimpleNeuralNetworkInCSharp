using NeuralNetworkCSharp;
using NeuralNetworkCSharp.ActivationFunctions;
using NeuralNetworkCSharp.InputFunctions;
using System;

namespace NeuralNetworkRunner
{
    class Program
    {
        const int samples = 10000;

        static void Main(string[] args)
        {
            var network = new SimpleNeuralNetwork(1);

            var layerFactory = new NeuralLayerFactory();

            network.AddLayer(layerFactory.CreateNeuralLayer(2, new RectifiedActivationFuncion(), new WeightedSumFunction()));

            network.AddLayer(layerFactory.CreateNeuralLayer(1, new SigmoidActivationFunction(0.4), new WeightedSumFunction()));

            double[][] expectedValues = new double[samples][];
            double[][] trainingValues = new double[samples][];

            for(int i = 0; i < samples; i++)
            {
                Random rng = new Random();
                Random rng2 = new Random();

                int val1 = rng.Next(rng.Next() % 1000);
                int val2 = rng2.Next(i % 900);

                expectedValues[i] = new double[] { (val1 + val2) % 2 };
                trainingValues[i] = new double[] { val1, val2 };
                Console.WriteLine($"val1: {val1} val2: {val2} sum: { (val1 + val2) % 2 }");
            }

            network.PushExpectedValues(expectedValues);



            network.Train(trainingValues, 5000);



            network.PushInputValues(new double[] { 1054, 54 });
            var outputs = network.GetOutput();

            Console.WriteLine($"network output: {string.Join(", ", outputs)}");
            Console.ReadKey();
        }
    }
}
